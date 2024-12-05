import argparse
import json
import torch
from trainer import TrainerDictionnary
import os
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

def main(args):

    config = json.load(open('config.json'))
    exp_dir = os.path.join(config['log_dir'], config['exp_name'])
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer_inst = TrainerDictionnary(json.load(open('config.json')), args.device)
    trainer_inst.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-d', '--device', default="cpu", type=str, help='device to use')
    args = parser.parse_args()

    main(args)