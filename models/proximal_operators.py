import torch
import torch.nn as nn
import torch.nn.functional as F

def choose_prox(nb_atoms, config):
    if config['prox_type'] == 'l1norm':
        return L1NormProx(nb_atoms, config['lambda_init'], config['groupsize'])
    elif config['prox_type'] == 'l0norm':
        return SmoothL0NormProx(nb_atoms, config['lambda_init'], config['order'])
    else:
        raise ValueError('Prox type is not valid')
    

class L1NormProx(nn.Module):

    def __init__(self, nb_atoms, lmbda, groupsize):
        super(L1NormProx, self).__init__()

        self.groupsize = groupsize
        self.soft_threshold = nn.Softshrink(lambd=lmbda)
        self.scaling_lmbda = nn.Parameter(torch.zeros((1, nb_atoms, 1, 1)))
        self.ones = torch.ones(nb_atoms, 1, groupsize, groupsize)

    def forward(self, x):
        scaling_lmbda = torch.exp(self.scaling_lmbda)
        if self.groupsize == 1:
            x = self.soft_threshold(x * scaling_lmbda) / scaling_lmbda
        else:
            x_norms = torch.sqrt(F.conv2d(x**2, self.ones, stride=self.groupsize, groups=x.shape[1]))
            x_norms = F.conv_transpose2d(x_norms, self.ones, stride=self.groupsize, groups=x.shape[1])
            x = x / (x_norms + 1e-8)
            new_norms = self.soft_threshold(x_norms * scaling_lmbda) / scaling_lmbda
            x = x * new_norms

        return x
    
    def potential(self, coeffs):
        lmbda = torch.exp(self.scaling_lmbda) * self.soft_threshold.lambd
        if self.groupsize == 1:
            return torch.sum(torch.abs(coeffs) * lmbda, dim=1, keepdim=True)
        else:
            x_norms = torch.sqrt(F.conv2d(coeffs**2, self.ones, stride=self.groupsize, groups=coeffs.shape[1]))
            return torch.sum(x_norms * lmbda, dim=1, keepdim=True)
    
    def extra_repr(self):
        return f"""groupsize={self.groupsize}"""
    
    def _apply(self, fn):
        self.ones = fn(self.ones)
        return super()._apply(fn)
    
    
class SmoothL0NormProx(nn.Module):

    def __init__(self, nb_atoms, lmbda, order):
        super(SmoothL0NormProx, self).__init__()

        self.lmbda = lmbda
        self.scaling_lmbda = nn.Parameter(torch.zeros((1, nb_atoms, 1, 1)))
        self.order = order

    def forward(self, x):
        lmbda = torch.exp(self.scaling_lmbda)*self.lmbda
        x = x * (1 - lmbda**self.order / (lmbda**self.order + torch.abs(x)**self.order))
        return x
    
    def potential(self, coeffs):
        "Numerical Approximation of the potential function"
        prox_x = torch.linspace(-coeffs.abs().max().sqrt()*1.1, coeffs.abs().max().sqrt()*1.1, 101).to(coeffs.device)
        prox_x = prox_x*torch.abs(prox_x)
        prox_x = prox_x.view(1, 1, 1, 101).repeat(1, coeffs.shape[1], 1, 1)
        prox_y = self.forward(prox_x).squeeze()
        prox_x = prox_x.squeeze()
        grad_y = prox_x - prox_y
            
        potential_x = prox_y
        potential_y = torch.cumsum(grad_y, 1) / len(grad_y)
        potential_y = potential_y - potential_y[:,potential_y.shape[1]//2, None]

        idx = torch.searchsorted(potential_x[None,...].repeat(coeffs.shape[0], 1, 1), \
                                 coeffs.view(coeffs.shape[0], coeffs.shape[1], -1))
        cost = torch.zeros_like(coeffs[:,0,...])
        lmbda = (torch.exp(self.scaling_lmbda)*self.lmbda).squeeze()
        for i in range(coeffs.shape[0]):
            for j in range(coeffs.shape[1]):
                cost[i] += (potential_y[j, idx[i,j]+1] - potential_y[j, idx[i,j]]).view(coeffs.shape[2:]) * \
                           (coeffs[i,j] - potential_x[j, idx[i,j]].view(coeffs.shape[2:])) / \
                           (potential_x[j, idx[i,j]+1] - potential_x[j, idx[i,j]]).view(coeffs.shape[2:])*lmbda[j]
                cost[i] += potential_y[j, idx[i,j]].view(coeffs.shape[2:])*lmbda[j]
        return cost
    
    def extra_repr(self):
        return f"""lambda={self.lmbda}, order={self.order}"""