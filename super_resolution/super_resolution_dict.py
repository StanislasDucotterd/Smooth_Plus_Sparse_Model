import os
import sys
import math
import time
import torch
import argparse
import torch.nn.functional as F
from matplotlib import pyplot as plt
sys.path.append('..')
from utilities import batch_PSNR
from models.dictionary import Dictionary
torch.set_grad_enabled(False)
torch.manual_seed(10)

#Hyperparameter search for best performance on validation set

def adjust_img(img):
    return img[:,:,40:-40,40:-40]

def HtH(x):
    x = F.conv2d(x, gaussian_kernel, stride=4)
    x = F.conv_transpose2d(x, gaussian_kernel, stride=4)
    return x

def test_hyperparameter(model, beta, lambda_):
    if convex: model.prox.soft_threshold.lambd = lambda_
    else: model.prox.lmbda = lambda_
    model.beta.data = torch.tensor(beta)
    psnr = 0.
    images = os.listdir(folder)
    for image in images:
        img = torch.tensor(plt.imread(folder + image)[None,None,:,:]).to(device)
        y = F.conv2d(img, gaussian_kernel, stride=4)
        y += torch.randn_like(y) * 0.01
        Hty = F.conv_transpose2d(y, gaussian_kernel, stride=4)
        pred = model.reconstruct(Hty, HtH)[0]
        if testing: torch.save(pred.cpu(), 'pred_dict/' + image.split('.')[0] + '.pth')
        psnr += batch_PSNR(adjust_img(pred), adjust_img(img), 1.) / len(images)
    return psnr

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-d', '--device', default="cpu", type=str, help='device to use')
device = parser.parse_args().device

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

exp_name = 'sigma_25/NCPR'
infos = torch.load('../trained_models/' + exp_name + '/checkpoint.pth', map_location='cpu', weights_only=True)
config = infos['config']
convex = config['prox_params']['prox_type'] == 'l1norm'
model = Dictionary(config['dict_params'], config['prox_params'])
model.load_state_dict(infos['state_dict'])
model = model.to(device)
model.eval()
model.tol = 1e-5

folder = 'validation/'

best_beta = 1e-2
best_lambda = 1e-3
gamma = 4.0
testing = False
psnrs = {}
tests = 0

best_psnr = 0.
while gamma > 1.025 or testing:
    betas = sorted(list(set([best_beta/gamma, best_beta, best_beta*gamma])))
    lambdas = sorted(list(set([best_lambda/gamma, best_lambda, best_lambda*gamma])))
    for lambda_ in lambdas:
        for beta in betas:
            if len(psnrs) > 0:
                prev_hyper = torch.tensor([*psnrs.keys()])
                prev_hyper[:,0] = (prev_hyper[:,0] - beta).abs() / beta
                prev_hyper[:,1] = (prev_hyper[:,1] - lambda_).abs() / lambda_
                if prev_hyper.sum(dim=1).min() < 1e-3:
                    continue
            start = time.time()
            tests += 1
            psnr = test_hyperparameter(model, beta, lambda_)
            print(f'Beta: {beta:.6}, Lambda: {lambda_:.6}, PSNR: {psnr:.4f}, it took {time.time()-start:.2f}s')
            psnrs[(beta, lambda_)] = psnr
    if max(psnrs.values()) > best_psnr + 1e-4:
        best_psnr = max(psnrs.values())
        best_beta, best_lambda = max(psnrs, key=psnrs.get)
    else:
        gamma = math.sqrt(gamma)
        print('New gamma:', gamma)
    testing = False

print('Super Resolution')
print('Tolerance:', model.tol)
print(exp_name)
print('Number of tests:', tests)
print(f'Best PSNR is {best_psnr:.6} with beta={best_beta:.6} and lambda={best_lambda:.6}')