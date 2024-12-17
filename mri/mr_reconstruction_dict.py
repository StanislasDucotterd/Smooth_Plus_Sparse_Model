import os
import sys
import math
import time
import torch
import argparse
sys.path.append('..')
from utilities import batch_PSNR, center_crop
from models.dictionary import Dictionary
torch.set_grad_enabled(False)

#Hyperparameter search for best performance on validation set

def HtH(x, mask):
    Htx = torch.fft.fft2(x, norm='ortho')*mask
    return torch.fft.ifft2(Htx, norm='ortho').real

def test_hyperparameter(model, beta, lambda_):
    if convex: model.prox.soft_threshold.lambd = lambda_
    else: model.prox.lmbda = lambda_
    model.beta.data = torch.tensor(beta)
    psnr = 0.
    for image in images:
        mask = torch.load(data_folder + image + '/mask.pt', weights_only=True).to(device)
        x_gt = torch.load(data_folder + image + '/x_crop.pt', weights_only=True).to(device)
        y = torch.load(data_folder + image + '/y.pt', weights_only=True).to(device)
        Hty = torch.fft.ifft2(y*mask, norm='ortho').real.type(torch.float32)
        pred = model.reconstruct(Hty, lambda x: HtH(x, mask))[0]
        if testing: torch.save(pred.cpu(), data_folder + image + '/pred_dict.pt')
        pred = center_crop(pred, (320, 320))
        psnr += batch_PSNR(pred, x_gt, 1.) / len(images)
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

exp_name = 'sigma_25/NCPR'
infos = torch.load('../trained_models/' + exp_name + '/checkpoint.pth', map_location='cpu', weights_only=True)
config = infos['config']
convex = config['prox_params']['prox_type'] == 'l1norm'
model = Dictionary(config['dict_params'], config['prox_params'])
model.load_state_dict(infos['state_dict'])
model = model.to(device)
model.eval()
model.tol = 1e-5

#Optimal parameters can be found in best_hyperparameters.txt
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

print(data_folder)
print('Tolerance:', model.tol)
print(exp_name)
print('Number of tests:', tests)
print(f'Best PSNR is {best_psnr:.6} with beta={best_beta:.6} and lambda={best_lambda:.6}')