"""
training
"""
import os
import random
import logging
import argparse
import numpy as np
import scipy.io as sio
from time import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset import MSRA_Dataset
from network import DenseNet


def init_parser():
    parser = argparse.ArgumentParser(description='3D Hand Pose Estimation')
    parser.add_argument('--batchSize', type=int,
                        default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers')
    parser.add_argument('--nepoch', type=int, default=60,
                        help='number of epochs to train for')
    parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    # CUDA_VISIBLE_DEVICES=0 python train.py
    parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='learning rate at t=0')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (SGD only)')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0005, help='weight decay (SGD only)')
    parser.add_argument('--learning_rate_decay', type=float,
                        default=1e-7, help='learning rate decay')
    parser.add_argument('--size', type=str, default='full',
                        help='how many samples do we load: small | full')
    parser.add_argument('--JOINT_NUM', type=int, default=21,
                        help='number of joints')
    parser.add_argument('--test_index', type=int, default=0,
                        help='test index for cross validation, range: 0~8')
    parser.add_argument('--save_root_dir', type=str,
                        default='results',  help='output folder')
    parser.add_argument('--model', type=str, default='',
                        help='model name for training resume')
    parser.add_argument('--optimizer', type=str, default='',
                        help='optimizer name for training resume')

    global opt
    opt = parser.parse_args()


def main():
    init_parser()
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    save_dir = os.path.join(opt.save_root_dir, subject_names[opt.test_index])

    try:
        os.mkdir(save_dir)
        print('create save dir')
    except:
        print('dir already exist!')

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                        filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
    logging.info('======================================================')
    # 1 load data
    train_data = MSRA_Dataset(root_path='./result', opt=opt, train=True)
    train_dataloder = DataLoader(
        train_data, batch_size=opt.batchsize, shuffle=True, num_workers=int(opt.workers))
    test_data = MSRA_Dataset(root_path='./result', opt=opt, train=False)
    test_dataloder = DataLoader(
        test_data, batch_size=opt.batchsize, shuffle=False, num_workers=int(opt.workers))
    print('#Train data:', len(train_data), '#Test data:', len(test_data))
    print(opt)

    # define model,loss and optimizer
    net = DenseNet(opt)

    net.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))
    net.cuda()
    print(net)

    criterion = nn.MSELoss(size_average=True).cuda()
    optimizer = optim.Adam(
        net.parameters(),
        lr=opt.learning_rate,
        betas=(0.5, 0.999),
        eps=1e-06
    )

    if opt.optimizer != '':
        optimizer.load_state_dict(torch.load(
            os.path.join(save_dir, opt.optimizer)))

    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    for epoch in range(opt.nepoch):
        pass


def train():
    pass


def evaluate():
    pass


def visualize():
    pass


if __name__ == "__main__":
    main()
