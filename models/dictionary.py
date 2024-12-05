import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.proximal_operators import choose_prox
from models.orthonormalize import bjorck
from torchdeq import get_deq

class Dictionary(nn.Module):

    def __init__(self, dict_params, prox_params):
        super().__init__()

        self.nb_atoms = dict_params['nb_atoms']
        self.nb_free_atoms = dict_params['nb_free_atoms']
        self.atom_size = dict_params['atom_size']
        self.tol = dict_params['tol']
        self.atoms = nn.Parameter(torch.randn(self.nb_atoms, 1, self.atom_size, self.atom_size) / 3000)
        self.free_atoms = nn.Parameter(torch.randn(self.nb_free_atoms, 1, self.atom_size, self.atom_size) / 3000)
        self.beta = nn.Parameter(torch.tensor(dict_params['beta_init']))
        self.prox = choose_prox(self.nb_atoms, prox_params)
        self.padding = (self.atom_size-1)//2

        self.dirac = torch.zeros(1, 1, self.atom_size*2-1, self.atom_size*2-1)
        self.dirac[0, 0, self.atom_size-1, self.atom_size-1] = 1.0

        self.b_solver = dict_params['b_solver']
        self.b_max_iter = dict_params['b_max_iter']

    def forward(self, y):

        # Compute the projection operator and the associated filter
        free_atoms = bjorck(self.free_atoms.view(self.nb_free_atoms, -1))
        proj = torch.eye(self.atom_size**2, device=y.device) - free_atoms.T @ free_atoms
        proj_filters = (proj - proj.mean(dim=1).repeat(self.atom_size**2, 1)).reshape(self.atom_size**2, 1, self.atom_size, self.atom_size)
        proj_impulse = self.conv_transpose2d(self.conv2d(self.dirac, proj_filters), proj_filters)

        # Compute the Lipschitz constant
        L_PktPk = torch.fft.fft2(proj_impulse, s=[y.shape[2], y.shape[3]]).abs().max()
        beta = F.relu(self.beta)
        L = 1.01*beta*L_PktPk
        
        #Project the atoms in the nullspace and normalize them
        atoms = self.atoms - torch.mean(self.atoms, dim=(1,2,3), keepdim=True)
        nullspace = torch.tensordot(atoms.view(self.nb_atoms, -1), free_atoms.T @ free_atoms, dims=([1], [0]))
        atoms = atoms - nullspace.view(self.nb_atoms, 1, self.atom_size, self.atom_size)
        atoms = atoms / torch.linalg.norm(atoms.view(self.nb_atoms, -1), axis=1, ord=2).view(-1, 1, 1, 1)
        atoms = torch.sqrt(0.99 / torch.maximum(beta, torch.tensor(0.1))) * atoms / (torch.linalg.matrix_norm(atoms.view(self.nb_atoms, -1), 2))
        X = (atoms.view(self.nb_atoms, -1) @ atoms.view(self.nb_atoms, -1).T)[...,None, None]

        # Initialization
        img = y.clone()
        coeffs = torch.zeros_like(self.conv2d(y, atoms))
        img_shape, img_size = img.shape, img[0].nelement()
        coeffs_shape = coeffs.shape

        # Define one step of PALM (iPALM without acceleration) for the DEQ
        def f(img, coeffs):
            coeffs = self.prox(coeffs - beta*(F.conv2d(coeffs, X) - self.conv2d(img, atoms)))
            dict_pred = self.conv_transpose2d(coeffs, atoms)
            sum_PktPk = self.conv2d(img, proj_impulse, padding=self.atom_size-1)
            img = img - (img - y + beta*(sum_PktPk - dict_pred)) / L
            return img, coeffs

        def ipalm_solver(deq_func, x0, max_iter, tol, stop_mode, **solver_kwargs):
            img, new_img = x0[:,:img_size].view(img_shape).clone(), x0[:,:img_size].view(img_shape).clone()
            coeffs, new_coeffs = x0[:,img_size:].view(coeffs_shape).clone(), x0[:,img_size:].view(coeffs_shape).clone()
            k, k_mean = 1, 0.
            # Once an image converged, it is not updated anymore
            idx = torch.arange(0, y.shape[0], device=y.device)
            res = torch.ones(y.shape[0], device=y.device)
            while max(res) > tol and k < max_iter:
                temp_coeffs = new_coeffs[idx] + (new_coeffs[idx] - coeffs[idx])*(k-1)/(k+2)
                coeffs[idx] = new_coeffs[idx].clone()
                new_coeffs[idx] = self.prox(temp_coeffs - beta*(F.conv2d(temp_coeffs, X) - self.conv2d(new_img[idx], atoms)))

                temp_img = new_img[idx] + (new_img[idx] - img[idx])*(k-1)/(k+2)
                dict_pred = self.conv_transpose2d(new_coeffs[idx], atoms)
                sum_PktPk = self.conv2d(temp_img, proj_impulse, padding=self.atom_size-1)
                img[idx] = new_img[idx].clone()
                new_img[idx] = temp_img - (temp_img - y[idx] + beta*(sum_PktPk - dict_pred)) / L

                res = (img - new_img).view(img.shape[0], -1).norm(dim=1) / img.view(img.shape[0], -1).norm(dim=1)
                idx = (res > tol).nonzero().view(-1)
                k_mean += torch.sum((res > tol)).item() / img.shape[0]
                k += 1
                if k == max_iter: print('Max iter reached')
                
            info = {}
            info['mean_steps'] = k_mean
            info['max_steps'] = k
            z_list = [new_img, new_coeffs]
            z_list = torch.cat([z.flatten(start_dim=1) if z.dim() >= 2 else z.view(z.nelement(), 1) for z in z_list], dim=1)
            return z_list, [], info
        
        deq = get_deq(f_max_iter=10000, f_tol=self.tol, b_solver=self.b_solver, \
                      b_max_iter=self.b_max_iter, b_tol=1e-6, ift=True, kwargs={'ls': True})
        deq.f_solver = ipalm_solver
        z_out, infos = deq(f, (img, coeffs))
        final_img, final_coeffs = z_out[-1]
        return final_img, final_coeffs, infos

    def reconstruct(self, Hty, HtH, op_norm=1.):

        # Compute the projection operator and the associated filter
        free_atoms = bjorck(self.free_atoms.view(self.nb_free_atoms, -1))
        proj = torch.eye(self.atom_size**2, device=Hty.device) - free_atoms.T @ free_atoms
        proj_filters = (proj - proj.mean(dim=1).repeat(self.atom_size**2, 1)).reshape(self.atom_size**2, 1, self.atom_size, self.atom_size)
        proj_impulse = self.conv_transpose2d(self.conv2d(self.dirac, proj_filters), proj_filters)

        #Compute the Lipschitz constant
        L_PktPk = torch.fft.fft2(proj_impulse, s=[Hty.shape[2], Hty.shape[3]]).abs().max()
        beta = F.relu(self.beta)
        L = 1.01*(beta*L_PktPk+op_norm**2)
        
        #Project the atoms in the nullspace and normalize them
        atoms = self.atoms - torch.mean(self.atoms, dim=(1,2,3), keepdim=True)
        nullspace = torch.tensordot(atoms.view(self.nb_atoms, -1), free_atoms.T @ free_atoms, dims=([1], [0]))
        atoms = atoms - nullspace.view(self.nb_atoms, 1, self.atom_size, self.atom_size)
        atoms = atoms / torch.linalg.norm(atoms.view(self.nb_atoms, -1), axis=1, ord=2).view(-1, 1, 1, 1)
        atoms = torch.sqrt(0.99 / beta) * atoms / (torch.linalg.matrix_norm(atoms.view(self.nb_atoms, -1), 2))
        X = (atoms.view(self.nb_atoms, -1) @ atoms.view(self.nb_atoms, -1).T)[...,None, None]

        img, new_img = Hty.clone(), Hty.clone()
        coeffs, new_coeffs = torch.zeros_like(self.conv2d(Hty, atoms)), torch.zeros_like(self.conv2d(Hty, atoms))
        k, k_mean = 1, 0.
        idx = torch.arange(0, Hty.shape[0], device=Hty.device)
        res = torch.ones(Hty.shape[0], device=Hty.device)
        while max(res) > self.tol and k < 10000:
            temp_coeffs = new_coeffs[idx] + (new_coeffs[idx] - coeffs[idx])*(k-1)/(k+2)
            coeffs[idx] = new_coeffs[idx].clone()
            new_coeffs[idx] = self.prox(temp_coeffs - beta*(F.conv2d(temp_coeffs, X) - self.conv2d(new_img[idx], atoms)))

            temp_img = new_img[idx] + (new_img[idx] - img[idx])*(k-1)/(k+2)
            dict_pred = self.conv_transpose2d(new_coeffs[idx], atoms)
            sum_PktPk = self.conv2d(temp_img, proj_impulse, padding=self.atom_size-1)
            img[idx] = new_img[idx].clone()
            new_img[idx] = torch.clip(temp_img - (HtH(temp_img) - Hty[idx] + beta*(sum_PktPk - dict_pred)) / L, 0.)
                    
            res = (img - new_img).view(img.shape[0], -1).norm(dim=1) / img.view(img.shape[0], -1).norm(dim=1)
            idx = (res > self.tol).nonzero().view(-1)
            k_mean += torch.sum((res > self.tol)).item() / img.shape[0]
            k += 1
        infos = {}
        infos['mean_steps'] = k_mean
        infos['max_steps'] = k

        return new_img, new_coeffs, infos
    
    def conv2d(self, x, filters, padding=None):
        if padding is None: padding = self.padding
        return F.conv2d(F.pad(x, pad=(padding,)*4, mode='circular'), filters)
    
    def conv_transpose2d(self, x, filters, padding=None):
        if padding is None: padding = self.padding
        return F.conv2d(F.pad(x, pad=(padding,)*4, mode='circular'), filters.flip(2, 3).permute(1, 0, 2, 3))

    def extra_repr(self):
        return f"""nb_atoms={self.nb_atoms},  nb_free_atoms={self.nb_free_atoms}, 
        atom_size={self.atom_size}, res={self.tol}"""
    
    def _apply(self, fn):
        self.dirac = fn(self.dirac)
        return super()._apply(fn)