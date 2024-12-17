import os
import sys
import time
import math
import torch
import argparse
import warnings
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
sys.path.append('/home/ducotter/ipalm_synthesis')
from utils import utilities
from utils.geometric_transform import center_crop
torch.manual_seed(10)
torch.set_grad_enabled(False)
warnings.filterwarnings("ignore")

def HtH(x):
    x = F.conv2d(x, gaussian_kernel, stride=4)
    x = F.conv_transpose2d(x, gaussian_kernel, stride=4)
    return x

def L_tv(x, filters): 
    filters1 = filters[0]; filters2 = filters[1]
    
    L1 = F.pad(F.conv2d(x, filters1), (0, 1), "constant", 0)
    L2 = F.pad(F.conv2d(x, filters2), (0, 0, 0, 1), "constant", 0)

    Lx = torch.cat((L1, L2), dim=1)

    return Lx 

def Lt_tv(y, filters): 
    filters1 = filters[0]; filters2 = filters[1]

    L1t = F.conv_transpose2d(y[:, 0:1, :, :-1], filters1)
    L2t = F.conv_transpose2d(y[:, 1:2, :-1, :], filters2)

    Lty = L1t + L2t

    return Lty

def prox_tv(y, niter, lmbda): 

    filters1 = torch.Tensor([[[[1., -1]]]]).to(device)
    filters2 = torch.Tensor([[[[1], [-1]]]]).to(device)
    filters = [filters1, filters2]

    v_k = torch.zeros((1, 2, y.shape[0], y.shape[1]), requires_grad=False, device=device)
    u_k = torch.zeros((1, 2, y.shape[0], y.shape[1]), requires_grad=False, device=device)

    t_k = 1
    alpha = 1/(8*lmbda)

    for _ in range(niter):
        Ltv = Lt_tv(v_k, filters)
        pc = torch.clip(y - lmbda * Ltv, 0)
        Lpc = L_tv(pc, filters)

        temp = v_k + alpha * Lpc
        
        u_kp1 = torch.nn.functional.normalize(temp, eps=1, dim=1, p=2)

        t_kp1 = (1 + np.sqrt(4*t_k**2+1)) / 2
        v_kp1 = u_kp1  + (t_k - 1) / t_kp1 * (u_kp1 - u_k)                                                                                                                                                                                                                                                                                                                                                         

        u_k = u_kp1
        v_k = v_kp1
        t_k = t_kp1

    Ltu = Lt_tv(u_k, filters)
    c = torch.clip(y - lmbda * Ltu, 0)

    return c  

def test_hyperparameter(lambda_):
    if testing:
        psnr = 0.
        ssim = 0.
        true_img = torch.tensor(plt.imread('img_test_material.png')[None,None,:,:]).to(device)
        y = F.conv2d(true_img, gaussian_kernel, stride=4) 
        y += torch.randn_like(y) * 0.01
        Hty = F.conv_transpose2d(y, gaussian_kernel, stride=4)
        x = Hty.clone()
        z = Hty.clone()
        k, t, res = 1, 1.0, 1.0
        while res > tol and k < 1000:
            x_old = torch.clone(x)
            x = z - HtH(z) + Hty
            x = prox_tv(x.squeeze(), tv_iter, lambda_)
            t_old = t 
            t = 0.5 * (1 + math.sqrt(1 + 4*t**2))
            z = x + (t_old - 1)/t * (x - x_old)
            res = (torch.norm(x_old - x)/torch.norm(x_old)).item()
            k += 1
        torch.save(x.cpu(), 'reco_tv.pth')
        aa
        for image in images:
            img = torch.tensor(plt.imread('test/' + image)[None,None,:,:]).to(device)
            y = F.conv2d(img, gaussian_kernel, stride=4)
            y += torch.randn_like(y) * 0.01
            Hty = F.conv_transpose2d(y, gaussian_kernel, stride=4)
            x = Hty.clone()
            z = Hty.clone()
            k, t, res = 1, 1.0, 1.0
            while res > tol and k < 1000:
                x_old = torch.clone(x)
                x = z - HtH(z) + Hty
                x = prox_tv(x.squeeze(), tv_iter, lambda_)
                t_old = t 
                t = 0.5 * (1 + math.sqrt(1 + 4*t**2))
                z = x + (t_old - 1)/t * (x - x_old)
                res = (torch.norm(x_old - x)/torch.norm(x_old)).item()
                k += 1
            psnr += utilities.batch_PSNR(x[:,:,40:-40,40:-40], img[:,:,40:-40,40:-40], 1.) / len(images)
            ssim += utilities.batch_SSIM(x[:,:,40:-40,40:-40], img[:,:,40:-40,40:-40], 1.) / len(images)
    else:
        true_img = torch.tensor(plt.imread('img_learn_material.png')[None,None,:,:]).to(device)
        y = F.conv2d(true_img, gaussian_kernel, stride=4) 
        y += torch.randn_like(y) * 0.01
        Hty = F.conv_transpose2d(y, gaussian_kernel, stride=4)
        x = Hty.clone()
        z = Hty.clone()
        k, t, res = 1, 1.0, 1.0
        while res > tol and k < 1000:
                x_old = torch.clone(x)
                x = z - HtH(z) + Hty
                x = prox_tv(x.squeeze(), tv_iter, lambda_)
                t_old = t 
                t = 0.5 * (1 + math.sqrt(1 + 4*t**2))
                z = x + (t_old - 1)/t * (x - x_old)
                res = (torch.norm(x_old - x)/torch.norm(x_old)).item()
                k += 1
        psnr = utilities.batch_PSNR(x[:,:,40:-40,40:-40], true_img[:,:,40:-40,40:-40], 1.)
    print('SSIM:', ssim)
    return psnr, ssim

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

images = os.listdir('test/')

tol = 1e-5
tv_iter = 50
best_lambda = 0.000310464
gamma = 1.0
testing = True
psnrs = {}
tests = 0

best_psnr = 0.
while gamma > 1.025 or testing:
    lambdas = sorted(list(set([best_lambda/gamma, best_lambda, best_lambda*gamma])))
    for lambda_ in lambdas:
        if len(psnrs) > 0:
            prev_hyper = torch.tensor([*psnrs.keys()])
            prev_hyper = (prev_hyper - lambda_).abs() / lambda_
            if prev_hyper.min() < 1e-3:
                continue
        start = time.time()
        tests += 1
        psnr = test_hyperparameter(lambda_)
        print(f'Lambda: {lambda_:.6}, PSNR: {psnr:.4f}, it took {time.time()-start:.2f}s')
        psnrs[lambda_] = psnr
    if max(psnrs.values()) > best_psnr + 1e-4:
        best_psnr = max(psnrs.values())
        best_lambda = max(psnrs, key=psnrs.get)
    else:
        gamma = math.sqrt(gamma)
        print('New gamma:', gamma)
    testing = False

print('TV iter', tv_iter)
print('Tolerance:', tol)
print('TV')
print('Number of tests:', tests)
print(f'Best PSNR is {best_psnr:.6} with lambda={best_lambda:.6}')