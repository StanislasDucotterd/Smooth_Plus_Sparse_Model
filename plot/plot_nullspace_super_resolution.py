import os
import sys
import torch
import warnings
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
sys.path.append('/home/ducotter/ipalm_synthesis')
from models.dictionary import Dictionary
torch.manual_seed(10)
torch.set_grad_enabled(False)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-d', '--device', default="cpu", type=str, help='device to use')
device = parser.parse_args().device

fig, ax = plt.subplots(2, 5, figsize=(6, 2.5), dpi=1000)

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

# Convex Model

true_img = torch.tensor(plt.imread('/home/ducotter/ipalm_synthesis/super_resolution/img_test_material.png')[None,None,:,:]).to(device)
y = F.conv2d(true_img, gaussian_kernel, stride=4) 
y += torch.randn_like(y) * 0.01
Hty = F.conv_transpose2d(y, gaussian_kernel, stride=4)

def HtH(x):
    x = F.conv2d(x, gaussian_kernel, stride=4)
    x = F.conv_transpose2d(x, gaussian_kernel, stride=4)
    return x

exp_name = 'sigma_25/size_13_200_atoms_120_free_atoms_st_2e-3_beta_2_groupsize_2_batch_16_sched_0.75'
infos = torch.load('/home/ducotter/ipalm_synthesis/exps/' + exp_name + '/checkpoints/checkpoint.pth', map_location='cpu')
config = infos['config']
model = Dictionary(config['dict_params'], config['prox_params'])
model.load_state_dict(infos['state_dict'], strict=False)
model = model.to(device)
model.eval()

model.tol = 1e-5
model.beta.data = torch.tensor(0.0389756)
model.prox.soft_threshold.lambd = 5.2556e-05

final_img, final_coeffs, _ = model.reconstruct(Hty, lambda x: HtH(x))
cost = model.prox.cost(final_coeffs)

model.prox.soft_threshold.lambd = 1e12
convex_nullspace_img, _, _ = model.reconstruct(Hty, lambda x: HtH(x))

ax[0,0].imshow(Hty.cpu().squeeze().numpy()[40:-40,40:-40], cmap='gray')
ax[0,0].set_title('Low-Res', fontsize=6)
ax[0,0].axis('off')
ax[0,1].imshow(final_img.cpu().squeeze().numpy()[40:-40,40:-40], cmap='gray')
ax[0,1].set_title('Prediction', fontsize=6)
ax[0,1].axis('off')
ax[0,2].imshow(convex_nullspace_img.cpu().squeeze().numpy()[40:-40,40:-40], cmap='gray')
ax[0,2].set_title('Free Image', fontsize=6)
ax[0,2].axis('off')
ax[0,3].imshow((final_img - convex_nullspace_img).cpu().squeeze().numpy()[40:-40,40:-40], cmap='gray')
ax[0,3].set_title('Regularized Image', fontsize=6)
ax[0,3].axis('off')
ax[0,4].imshow(cost.cpu().squeeze().numpy()[40:-40,40:-40], cmap='gray')
ax[0,4].set_title('Cost', fontsize=6)
ax[0,4].axis('off')
fig.text(0.0075, 0.69, 'CPR', va='center', rotation='vertical', fontsize=6)

#  Nonconvex Model

exp_name = 'sigma_25/size_13_200_atoms_120_free_atoms_order_1.8_ht_1.25e-2_beta_2_groupsize_1_batch_16_sched_0.9'
infos = torch.load('/home/ducotter/ipalm_synthesis/exps/' + exp_name + '/checkpoints/checkpoint.pth', map_location='cpu')
config = infos['config']
model = Dictionary(config['dict_params'], config['prox_params'])
model.load_state_dict(infos['state_dict'], strict=False)
model = model.to(device)
model.eval()

model.tol = 1e-5
model.beta.data = torch.tensor(0.00258673)
model.prox.lmbda = 0.000929068

final_img, final_coeffs, _ = model.reconstruct(Hty, lambda x: HtH(x))
cost = model.prox.cost(final_coeffs)

model.prox.lmbda = 1e12
nullspace_img, _, _ = model.reconstruct(Hty, lambda x: HtH(x))

ax[1,0].set_ylabel('NCPR', fontsize=6)
ax[1,0].imshow(Hty.cpu().squeeze().numpy()[40:-40,40:-40], cmap='gray')
ax[1,0].axis('off')
ax[1,1].imshow(final_img.cpu().squeeze().numpy()[40:-40,40:-40], cmap='gray')
ax[1,1].axis('off')
ax[1,2].imshow(nullspace_img.squeeze().cpu()[40:-40,40:-40], cmap='gray')
ax[1,2].axis('off')
ax[1,3].imshow((final_img - nullspace_img).cpu().squeeze().numpy()[40:-40,40:-40], cmap='gray')
ax[1,3].axis('off')
ax[1,4].imshow(cost.cpu().squeeze().numpy()[40:-40,40:-40], cmap='gray')
ax[1,4].axis('off')
fig.text(0.0075, 0.265, 'NCPR', va='center', rotation='vertical', fontsize=6)


print('MSE of convex nullspace and nonconvex nullspace: ', ((convex_nullspace_img - nullspace_img)**2).mean())

plt.tight_layout()
plt.savefig('decomposition_superres.pdf')