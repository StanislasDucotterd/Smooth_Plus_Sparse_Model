import torch
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from BSD500 import BSD500
from models.dictionary import Dictionary
from utilities import batch_PSNR, batch_SSIM

class TrainerDictionnary:
    """
    """
    def __init__(self, config, device):

        self.config = config
        self.device = device
        self.sigma = config['sigma']

        # Prepare dataset classes
        train_dataset = BSD500(config['training_options']['train_data_file'], config['prox_params']['groupsize'])
        val_dataset = BSD500(config['training_options']['val_data_file'], config['prox_params']['groupsize'])

        print('Preparing the dataloaders')
        self.train_dataloader = DataLoader(train_dataset, batch_size=config["training_options"]["batch_size"], shuffle=True,\
                                             num_workers=config["training_options"]["num_workers"], drop_last=True)

        self.val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
        self.batch_size = config["training_options"]["batch_size"]

        print('Building the model')
        self.model = Dictionary(config['dict_params'], config['prox_params'])
        self.model = self.model.to(device)
        print(self.model)
        
        self.epochs = config["training_options"]['epochs']
        params = [{'params': [self.model.atoms, self.model.free_atoms], 'lr': config['training_options']['lr_atoms']}, \
                  {'params': list(self.model.prox.parameters()) + [self.model.beta], 'lr': config['training_options']['lr_hyper']}]

        self.optimizer = torch.optim.Adam(params)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config['training_options']['lr_decay'])
        self.criterion = torch.nn.L1Loss()

        # CHECKPOINTS & TENSOBOARD
        run_name = config['exp_name']
        print('Run name: ', run_name)
        self.checkpoint_dir = os.path.join(config['log_dir'], config["exp_name"])
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        config_save_path = os.path.join(config['log_dir'], config["exp_name"], 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        writer_dir = os.path.join(config['log_dir'], config["exp_name"], 'tensorboard_logs')
        self.writer = tensorboard.SummaryWriter(writer_dir)

        self.total_training_step = 1
        self.total_eval_step = 0
        

    def train(self):
        
        for epoch in range(self.epochs+1):
            self.train_epoch(epoch)

        self.writer.flush()
        self.writer.close()

    def train_epoch(self, epoch):

        self.model.train()
        tbar = tqdm(self.train_dataloader, ncols=135, position=0, leave=True)
        log = {}
        for idx, data in enumerate(tbar):
            image = data.to(self.device)
            noisy_image = image + (self.sigma/255.0)*torch.randn(data.shape, device=self.device)

            self.optimizer.zero_grad()
            output, coeffs, infos = self.model(noisy_image)

            loss = self.criterion(output, image)
            loss.backward()
            self.optimizer.step()
                
            log['beta'] = self.model.beta.item()
            log['Median coeff amplitude'] = torch.median(torch.abs(coeffs)).cpu().item()
            log['95% percentile coeff amplitude'] = torch.quantile(torch.abs(coeffs), 0.95).cpu().item()
            log['train_loss'] = loss.detach().cpu().item()
            log['mean_steps'] = infos['mean_steps']
            log['max_steps'] = infos['max_steps']

            self.wrt_step = self.total_training_step * self.batch_size
            self.write_scalars_tb(log)

            if self.total_training_step % max((len(tbar) // 10), 1) == 0:
                self.valid_epoch()
                self.scheduler.step()
                self.save_checkpoint('/checkpoint')

            tbar.set_description('T ({}) | TotalLoss {:.5f} |'.format(epoch, log['train_loss'])) 
            self.total_training_step += 1

    def valid_epoch(self):
        
        self.model.eval()
        loss_val, psnr_val, ssim_val = 0.0, 0.0, 0.0
        tbar_val = tqdm(self.val_dataloader, ncols=130, position=0, leave=True)
        with torch.no_grad():
            for batch_idx, data in enumerate(tbar_val):
                image = data.to(self.device)
                noisy_image = image + (self.sigma/255.0)*torch.randn(image.shape, device=self.device)
                output, _, _ = self.model(noisy_image)
                loss_val += self.criterion(output, image).cpu().item()/self.val_dataloader.__len__()
                output = torch.clamp(output, 0., 1.)
                psnr_val += batch_PSNR(output, image, 1.)/self.val_dataloader.__len__()
                ssim_val += batch_SSIM(output, image, 1.)/self.val_dataloader.__len__()
        

        self.writer.add_image('my_image', output[0,0,...], self.total_eval_step, dataformats='HW')
        self.writer.add_scalar(f'val/loss', loss_val, self.total_eval_step)
        self.writer.add_scalar(f'val/Test PSNR Mean', psnr_val, self.total_eval_step)
        self.writer.add_scalar(f'val/Test SSIM Mean', ssim_val, self.total_eval_step)

        self.total_eval_step += 1
        self.model.train()


    def write_scalars_tb(self, logs):
        for k, v in logs.items():
            self.writer.add_scalar(f'train/{k}', v, self.wrt_step)

    def save_checkpoint(self, name):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }

        print('Saving a checkpoint:')
        filename = self.checkpoint_dir + name + '.pth'
        torch.save(state, filename)