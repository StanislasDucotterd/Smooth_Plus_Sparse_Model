import os
import sys
import time
import math
import torch
import argparse
import numpy as np
import torch.nn.functional as F
sys.path.append('..')
from utilities import batch_PSNR, center_crop
torch.set_grad_enabled(False)

#Hyperparameter search for best performance on validation set

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

    filters1 = torch.Tensor([[[[1., -1]]]]).to(device).double()
    filters2 = torch.Tensor([[[[1], [-1]]]]).to(device).double()
    filters = [filters1, filters2]

    v_k = torch.zeros((1, 2, y.shape[0], y.shape[1]), requires_grad=False, device=device).double()
    u_k = torch.zeros((1, 2, y.shape[0], y.shape[1]), requires_grad=False, device=device).double()

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
    psnr = 0.
    for image in images:
        mask = torch.load(data_folder + image + '/mask.pt', weights_only=True).to(device)
        x_gt = torch.load(data_folder + image + '/x_crop.pt', weights_only=True).to(device)
        y = torch.load(data_folder + image + '/y.pt', weights_only=True).to(device)
        Hty = torch.fft.ifft2(y*mask, norm='ortho').real.type(torch.float32)
        x = Hty.clone()
        z = Hty.clone()
        k, t, res = 1, 1.0, 1.0
        while res > tol and k < 1000:
            x_old = torch.clone(x)
            x = z - torch.fft.ifft2(torch.fft.fft2(z, norm='ortho')*mask, norm='ortho').real + Hty
            x = prox_tv(x.squeeze(), tv_iter, lambda_)
            t_old = t 
            t = 0.5 * (1 + math.sqrt(1 + 4*t**2))
            z = x + (t_old - 1)/t * (x - x_old)
            res = (torch.norm(x_old - x)/torch.norm(x_old)).item()
            k += 1
        if testing: torch.save(x.cpu(), data_folder + image + '/pred_tvb.pt')
        psnr += batch_PSNR(center_crop(x, (320, 320)), x_gt, 1.) / len(images)
    return psnr

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-d', '--device', default="cpu", type=str, help='device to use')
device = parser.parse_args().device

data_folder = 'acc_8_cf_0.04/pd/val_images/'
images = os.listdir(data_folder)

psnr = 0.
for image in images:
    mask = torch.load(data_folder + image + '/mask.pt', weights_only=True).to(device)
    x_gt = torch.load(data_folder + image + '/x_crop.pt', weights_only=True).to(device)
    y = torch.load(data_folder + image + '/y.pt', weights_only=True).to(device)
    Hty = torch.fft.ifft2(y*mask, norm='ortho').real.type(torch.float32)
    psnr += batch_PSNR(center_crop(Hty, (320, 320)), x_gt, 1.) / len(images)
print(f'PSNR of the zero-filled reconstruction is {psnr:.6}')

tol = 1e-5
tv_iter = 50
best_lambda = 1e-2
gamma = 4.0
testing = False
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

print(data_folder)
print('TV iter', tv_iter)
print('Tolerance:', tol)
print('Number of tests:', tests)
print(f'Best PSNR is {best_psnr:.6} with lambda={best_lambda:.6}')