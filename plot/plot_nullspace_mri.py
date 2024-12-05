import os
import sys
import torch
import warnings
import argparse
import matplotlib.pyplot as plt
sys.path.append('/home/ducotter/ipalm_synthesis')
from models.dictionary import Dictionary
from utils.geometric_transform import center_crop
torch.manual_seed(10)
torch.set_grad_enabled(False)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-d', '--device', default="cpu", type=str, help='device to use')
device = parser.parse_args().device

fig, ax = plt.subplots(2, 5, figsize=(6, 2.5), dpi=1000)

# Convex Model

path = '/home/ducotter/ipalm_synthesis/mri/data_sets/singlecoil_acc_8_cf_0.04_noisesd_0.002/pd/test_images/'
images = os.listdir(path)
image = images[9]
print(image)

mask = torch.load(path + image + '/mask.pt').to(device)
y = torch.load(path + image + '/y.pt').to(device)
Hty = torch.fft.ifft2(y*mask, norm='ortho').real

def HtH(x, mask):
    Htx = torch.fft.fft2(x, norm='ortho')*mask
    return torch.fft.ifft2(Htx, norm='ortho').real

exp_name = 'sigma_25/size_13_200_atoms_120_free_atoms_st_2e-3_beta_2_groupsize_2_batch_16_sched_0.75'
infos = torch.load('/home/ducotter/ipalm_synthesis/exps/' + exp_name + '/checkpoints/checkpoint.pth', map_location='cpu')
config = infos['config']
model = Dictionary(config['dict_params'], config['prox_params'])
model.load_state_dict(infos['state_dict'], strict=False)
model = model.to(device)
model.eval()

model.tol = 1e-5
# Acceleration 8 - PD
# model.beta.data = torch.tensor(0.480868)
# model.prox.soft_threshold.lambd = 0.000272627
# Acceleration 16 - PDFS
model.beta.data = torch.tensor(0.440958)
model.prox.lmbda = 0.00025

final_img, final_coeffs, _ = model.reconstruct(Hty, lambda x: HtH(x, mask))
cost = model.prox.cost(final_coeffs)

model.prox.soft_threshold.lambd = 1e12
convex_nullspace_img, _, _ = model.reconstruct(Hty, lambda x: HtH(x, mask))

ax[0,0].imshow(center_crop(Hty.cpu().squeeze().numpy(), (320, 320)), cmap='gray')
ax[0,0].set_title('Zero-fill', fontsize=6)
ax[0,0].axis('off')
ax[0,1].imshow(center_crop(final_img.cpu().squeeze().numpy(), (320, 320)), cmap='gray')
ax[0,1].set_title('Prediction', fontsize=6)
ax[0,1].axis('off')
ax[0,2].imshow(center_crop(convex_nullspace_img.cpu().squeeze().numpy(), (320, 320)), cmap='gray')
ax[0,2].set_title('Free Image', fontsize=6)
ax[0,2].axis('off')
ax[0,3].imshow(center_crop((final_img - convex_nullspace_img).cpu().squeeze().numpy(), (320, 320)), cmap='gray')
ax[0,3].set_title('Regularized Image', fontsize=6)
ax[0,3].axis('off')
ax[0,4].imshow(center_crop(cost.cpu().squeeze().numpy(), (160, 160)), cmap='gray')
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
# Acceleration 8 - PD
# model.beta.data = torch.tensor(0.0189763)
# model.prox.lmbda = 0.0013139
# Acceleration 16 - PDFS
model.beta.data = torch.tensor(0.015)
model.prox.lmbda = 0.001

final_img, final_coeffs, _ = model.reconstruct(Hty, lambda x: HtH(x, mask))
cost = model.prox.cost(final_coeffs)

model.prox.lmbda = 1e12
nullspace_img, _, _ = model.reconstruct(Hty, lambda x: HtH(x, mask))

ax[1,0].set_ylabel('NCPR', fontsize=6)
ax[1,0].imshow(center_crop(Hty.cpu().squeeze().numpy(), (320, 320)), cmap='gray')
ax[1,0].axis('off')
ax[1,1].imshow(center_crop(final_img.cpu().squeeze().numpy(), (320, 320)), cmap='gray')
ax[1,1].axis('off')
ax[1,2].imshow(center_crop(nullspace_img.cpu().squeeze().numpy(), (320, 320)), cmap='gray')
ax[1,2].axis('off')
ax[1,3].imshow(center_crop((final_img - nullspace_img).cpu().squeeze().numpy(), (320, 320)), cmap='gray')
ax[1,3].axis('off')
ax[1,4].imshow(center_crop(cost.cpu().squeeze().numpy(), (320, 320)), cmap='gray')
ax[1,4].axis('off')
fig.text(0.0075, 0.265, 'NCPR', va='center', rotation='vertical', fontsize=6)


print('MSE of convex nullspace and nonconvex nullspace: ', ((convex_nullspace_img - nullspace_img)**2).mean())

plt.tight_layout()
plt.savefig('decomposition_acc_16_pdfs_5.pdf')