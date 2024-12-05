import os 
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
sys.path.append('..')
from BSD500 import BSD500
from models.orthonormalize import bjorck
torch.set_grad_enabled(False)

exp_name = 'sigma_25/NCPR'
infos = torch.load('../trained_models/' + exp_name + '/checkpoint.pth', map_location='cpu', weights_only=True)
config = infos['config']
weights = infos['state_dict']
nb_atoms = config['dict_params']['nb_atoms']
nb_free_atoms = config['dict_params']['nb_free_atoms']
atom_size = config['dict_params']['atom_size']
lambdas = weights['prox.scaling_lmbda']
convex = config['prox_params']['prox_type'] == 'l1norm'

_, indices = torch.sort(lambdas, dim=1, descending=convex)
indices = indices.squeeze()

nb_x_atoms = 10
nb_y_atoms = 20
nb_x_fatoms = 10
nb_y_fatoms = 12

atoms = weights['atoms'][indices]
free_atoms = weights['free_atoms']
free_atoms = bjorck(free_atoms.view(nb_free_atoms, -1))

plt.style.use('dark_background')

atoms = atoms - torch.mean(atoms, dim=(1,2,3), keepdim=True)
scalar_products = atoms.view(nb_atoms, -1) @ free_atoms.view(nb_free_atoms, -1).T
nullspace = torch.tensordot(scalar_products, free_atoms, dims=([1], [0])).view(nb_atoms, atom_size**2, -1)
atoms = atoms - nullspace.view(nb_atoms, 1, atom_size, atom_size)
atoms = atoms / torch.linalg.norm(atoms.view(nb_atoms, -1), axis=1, ord=2).view(-1, 1, 1, 1)

mins = atoms.view(nb_atoms, -1).min(dim=1)[0].reshape(-1, 1, 1, 1)
maxs = atoms.view(nb_atoms, -1).max(dim=1)[0].reshape(-1, 1, 1, 1)
atoms = (atoms - mins) / (maxs - mins)

fig, ax = plt.subplots(nb_x_atoms, nb_y_atoms, figsize=(nb_y_atoms, nb_x_atoms))
for i in range(nb_x_atoms):
    for j in range(nb_y_atoms):
        ax[i,j].imshow(atoms[i*nb_y_atoms + j,0,:,:].squeeze(), cmap='gray')
        ax[i,j].axis('off')
plt.subplots_adjust(wspace=0.025, hspace=0.025)
plt.savefig('temporary_atoms.png', bbox_inches='tight', pad_inches=0)

free_atoms = free_atoms.view(nb_free_atoms, 1, atom_size, atom_size)
nullspace_stats = torch.zeros(nb_free_atoms, 1)
dataset = BSD500('/home/ducotter/nerf_dip/images/test.h5')
with torch.no_grad():
    for i in range(68):
        img = dataset.__getitem__(i).unsqueeze(0)
        proj_img = F.conv2d(img, free_atoms).squeeze().view(nb_free_atoms, -1)
        nullspace_stats = torch.cat((nullspace_stats, proj_img), dim=1)
nullspace_stats = nullspace_stats[:,1:]

cov = torch.cov(nullspace_stats)
L, V = torch.linalg.eig(cov)

fig, ax = plt.subplots(nb_x_fatoms, nb_y_fatoms, figsize=(nb_y_fatoms, nb_x_fatoms))
for i in range(nb_x_fatoms):
    for j in range(nb_y_fatoms):
        free_atom = (V[:,i*nb_y_fatoms + j].view(-1, 1, 1, 1)*free_atoms).sum(dim=(0,1)).real.cpu()
        free_atom = (free_atom - free_atom.min()) / (free_atom.max() - free_atom.min())
        ax[i,j].imshow(free_atom, cmap='gray')
        ax[i,j].axis('off')
plt.subplots_adjust(wspace=0.025, hspace=0.025)
plt.savefig('temporary_free_atoms.png', bbox_inches='tight', pad_inches=0)

free_atoms = Image.open('temporary_free_atoms.png')
free_atoms = np.array(free_atoms)[:,:,0]
atoms = Image.open('temporary_atoms.png')
atoms = np.array(atoms)[:,:,0]

plt.style.use('default')
fig, ax = plt.subplots(1, 2, width_ratios=[1, 5/3], figsize=(6, 2.25), dpi=2000)
ax[0].imshow(free_atoms, cmap='gray')
ax[0].axis('off')
ax[0].set_title('Free Atoms ($\mathbf{Q}$)')
ax[1].imshow(atoms, cmap='gray')
ax[1].axis('off')
ax[1].set_title('Regularized Atoms ($\mathbf{D}$)')
plt.tight_layout()
plt.savefig('plot_atoms.pdf')
os.remove('temporary_free_atoms.png')
os.remove('temporary_atoms.png')