import sys
import torch
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
sys.path.append('..')
from models.dictionary import Dictionary
from utilities import center_crop
torch.manual_seed(10)
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-d', '--device', default="cpu", type=str, help='device to use')
device = parser.parse_args().device

problem = 'denoising'
mri_type = 'acc_8'

def adjust_img(img):
    if problem == 'denoising':
        return img.squeeze().cpu()
    if problem == 'super_resolution':
        return img.squeeze()[40:-40,40:-40].cpu()
    if problem == 'mri':
        return center_crop(img, (320, 320)).squeeze().cpu()

if problem == 'denoising':
    img = torch.load('example_img.pth', weights_only=True)
    img = img.to(device)
    Hty = img + torch.randn_like(img)*25/255
    def HtH(x): return x
elif problem == 'super_resolution':
    kernel_size = 16
    sigma = 2.
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1)/2.
    gaussian_kernel = (1./(2.*torch.pi*sigma**2))*torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1)/(2*sigma**2))
    gaussian_kernel = (gaussian_kernel / torch.sum(gaussian_kernel)).view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(device)

    img = torch.tensor(plt.imread('img_test_material.png')[None,None,:,:]).to(device)
    y = F.conv2d(img, gaussian_kernel, stride=4) 
    y += torch.randn_like(y) * 0.01
    Hty = F.conv_transpose2d(y, gaussian_kernel, stride=4)

    def HtH(x):
        x = F.conv2d(x, gaussian_kernel, stride=4)
        x = F.conv_transpose2d(x, gaussian_kernel, stride=4)
        return x
elif problem == 'mri':
    mask = torch.load(mri_type + '/mask.pt', weights_only=True).to(device)
    y = torch.load(mri_type + '/y.pt', weights_only=True).to(device)
    Hty = torch.fft.ifft2(y*mask, norm='ortho').real

    def HtH(x):
        Htx = torch.fft.fft2(x, norm='ortho')*mask
        return torch.fft.ifft2(Htx, norm='ortho').real
else:
    raise ValueError('Unknown problem')


fig, ax = plt.subplots(2, 5, figsize=(6, 2.5), dpi=1000)

# Convex Model
exp_name = 'sigma_25/CPR'
infos = torch.load('../trained_models/' + exp_name + '/checkpoint.pth', map_location='cpu', weights_only=True)
config = infos['config']
model = Dictionary(config['dict_params'], config['prox_params'])
model.load_state_dict(infos['state_dict'])
model.to(device)
model.eval()

if problem == 'super_resolution':
    model.beta.data = torch.tensor(0.0389756)
    model.prox.soft_threshold.lambd = 5.2556e-05
    model.tol = 1e-5
if problem == 'mri': 
    model.tol = 1e-5
    if mri_type == 'acc_8':
        model.beta.data = torch.tensor(0.480868)
        model.prox.soft_threshold.lambd = 0.000272627
    if mri_type == 'acc_16':
        model.beta.data = torch.tensor(0.440958)
        model.prox.soft_threshold.lambd = 0.00025

final_img, final_coeffs, _ = model.reconstruct(Hty, HtH)
cost = model.prox.potential(final_coeffs)
model.prox.soft_threshold.lambd = 1e12
convex_nullspace_img, _, _ = model.reconstruct(Hty, HtH)

ax[0,0].imshow(adjust_img(Hty), cmap='gray')
ax[0,0].set_title('Input', fontsize=6)
ax[0,0].axis('off')
ax[0,1].imshow(adjust_img(final_img), cmap='gray')
ax[0,1].set_title('Prediction', fontsize=6)
ax[0,1].axis('off')
ax[0,2].imshow(adjust_img(convex_nullspace_img), cmap='gray')
ax[0,2].set_title('Free Image', fontsize=6)
ax[0,2].axis('off')
ax[0,3].imshow(adjust_img(final_img - convex_nullspace_img), cmap='gray')
ax[0,3].set_title('Regularized Image', fontsize=6)
ax[0,3].axis('off')
if problem == 'mri':
    ax[0,4].imshow(center_crop(cost, (160, 160)).squeeze().cpu(), cmap='gray')
else:
    ax[0,4].imshow(adjust_img(cost), cmap='gray')
ax[0,4].set_title('Cost', fontsize=6)
ax[0,4].axis('off')
fig.text(0.0075, 0.69, 'CPR', va='center', rotation='vertical', fontsize=6)

# Nonconvex Model
exp_name = 'sigma_25/NCPR'
infos = torch.load('../trained_models/' + exp_name + '/checkpoint.pth', map_location='cpu', weights_only=True)
config = infos['config']
model = Dictionary(config['dict_params'], config['prox_params'])
model.load_state_dict(infos['state_dict'])
model.to(device)
model.eval()

if problem == 'super_resolution':
    model.beta.data = torch.tensor(0.00258673)
    model.prox.lmbda = 0.000929068
    model.tol = 1e-5
if problem == 'mri': 
    model.tol = 1e-5
    if mri_type == 'acc_8':
        model.beta.data = torch.tensor(0.0189763)
        model.prox.lmbda = 0.0013139
    if mri_type == 'acc_16':
        model.beta.data = torch.tensor(0.015)
        model.prox.lmbda = 0.001

final_img, final_coeffs, _ = model.reconstruct(Hty, HtH)
cost = model.prox.potential(final_coeffs)
model.prox.lmbda = 1e12
nullspace_img, _, _ = model.reconstruct(Hty, HtH)

ax[1,0].set_ylabel('NCPR', fontsize=6)
ax[1,0].imshow(adjust_img(Hty), cmap='gray')
ax[1,0].axis('off')
ax[1,1].imshow(adjust_img(final_img), cmap='gray')
ax[1,1].axis('off')
ax[1,2].imshow(adjust_img(nullspace_img), cmap='gray')
ax[1,2].axis('off')
ax[1,3].imshow(adjust_img(final_img - nullspace_img), cmap='gray')
ax[1,3].axis('off')
ax[1,4].imshow(adjust_img(cost), cmap='gray')
ax[1,4].axis('off')
fig.text(0.0075, 0.265, 'NCPR', va='center', rotation='vertical', fontsize=6)

plt.tight_layout()
plt.savefig('decomposition.pdf')