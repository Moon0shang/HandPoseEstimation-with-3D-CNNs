"""
training
"""
import argparse
import os
import random
import logging
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from time import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

from dataset import TsdfDataset
from network import DenseNet

parser = argparse.ArgumentParser()
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
parser.add_argument('--SAMPLE_NUM', type=int, default=1024,
                    help='number of sample points')
parser.add_argument('--JOINT_NUM', type=int, default=21,
                    help='number of joints')
parser.add_argument('--INPUT_FEATURE_NUM', type=int,
                    default=6,  help='number of input point features')
parser.add_argument('--PCA_SZ', type=int, default=42,
                    help='number of PCA components')
parser.add_argument('--knn_K', type=int, default=64,  help='K for knn search')
parser.add_argument('--sample_num_level1', type=int,
                    default=512,  help='number of first layer groups')
parser.add_argument('--sample_num_level2', type=int,
                    default=128,  help='number of second layer groups')
parser.add_argument('--ball_radius', type=float, default=0.015,
                    help='square of radius for ball query in level 1')
parser.add_argument('--ball_radius2', type=float, default=0.04,
                    help='square of radius for ball query in level 2')

parser.add_argument('--test_index', type=int, default=0,
                    help='test index for cross validation, range: 0~8')
parser.add_argument('--save_root_dir', type=str,
                    default='results',  help='output folder')
parser.add_argument('--model', type=str, default='',
                    help='model name for training resume')
parser.add_argument('--optimizer', type=str, default='',
                    help='optimizer name for training resume')

opt = parser.parse_args()
print(opt)
# 在训练开始时，参数的初始化是随机的，为了让每次的结果一致，我们需要设置随机种子。
opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

save_dir = os.path.join(opt.save_root_dir, subject_names[opt.test_index])

try:
    os.makedirs(save_dir)
except:
    pass

logging.basicConfig
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                    filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
logging.info('======================================================')
# 1. Load data
train_data = HandPointDataset(
    root_path='D:\Cache\Git\HandPointNet\preprocess', opt=opt, train=True)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
                                               shuffle=True, num_workers=int(opt.workers), pin_memory=False)

test_data = HandPointDataset(
    root_path='D:\Cache\Git\HandPointNet\preprocess', opt=opt, train=False)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers), pin_memory=False)

print('#Train data:', len(train_data), '#Test data:', len(test_data))
print(opt)

# 2. Define model, loss and optimizer
net = DenseNet(opt)

net.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))

net.cuda()
print(net)

criterion = nn.MSELoss(size_average=True).cuda()
optimizer = optim.Adam(
    net.parameters(),
    lr=opt.learning_rate,
    betas == (0.5, 0.999),
    eps=1e-06
)
if opt.optimizer != '':
    optimizer.load_state_dict(torch.load(
        os.path.join(save_dir, opt.optimizer)))
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# training and testing
for epoch in range(opt.nepoch):
    scheduler.step(epoch)
    print('======>>>>> Online epoch: #%d, lr=%f, Test: %s <<<<<======' %
          (epoch, scheduler.get_lr()[0], subject_names[opt.test_index]))
    # switch to train mode
    torch.cuda.synchronize()
    net.train()
    train_mse = 0.0
    train_mse_wld = 0.0
    timer = time()

    for i, data in enumerate(tqdm(train_dataloader, 0)):
        if len(data[0]) == 1:
            continue
        torch.cuda.synchronize()
        # load input and target
        tsdf, ground_truth = data
        ground_truth = Variable(ground_truth, requires_grad=False).cuda()
        tsdf = tsdf.cuda()
        data_input = Variable(tsdf, requires_grad=False)

        # compute output
        optimizer.zero_grad()
        estimation = net(tsdf)
        loss = criterion(estimation, ground_truth)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        # update training error
        train_mse = train_mse + loss.data[0]

    # time taken
    torch.cuda.synchronize()
    timer = time() - timer
    timer = timer / len(train_data)

    print('==> time to learn 1 sample = %f (ms)' % (timer*1000))

    # print mse
    train_mse = train_mse / len(train_data)
    train_mse_wld = train_mse_wld / len(train_data)
    print('mean-square error of 1 sample: %f, #train_data = %d' %
          (train_mse, len(train_data)))
    print('average estimation error in world coordinate system: %f (mm)' %
          (train_mse_wld))

    torch.save(netR.state_dict(), '%s/netR_%d.pth' % (save_dir, epoch))
    torch.save(optimizer.state_dict(), '%s/optimizer_%d.pth' %
               (save_dir, epoch))

    net.eval()
    train_mse = 0.0
    train_mse_wld = 0.0
    timer = time()

    for i, data in enumerate(tqdm(train_dataloader, 0)):
        if len(data[0]) == 1:
            continue
        torch.cuda.synchronize()
        # load input and target
        tsdf, ground_truth = data
        ground_truth = Variable(ground_truth, requires_grad=False).cuda()
        tsdf = tsdf.cuda()
        data_input = Variable(tsdf, requires_grad=False)

        # compute output
        optimizer.zero_grad()
        estimation = net(tsdf)
        loss = criterion(estimation, ground_truth)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        # update training error
        train_mse = train_mse + loss.data[0]

    # time taken
    torch.cuda.synchronize()
    timer = time() - timer
    timer = timer / len(train_data)

    print('==> time to learn 1 sample = %f (ms)' % (timer*1000))

    # print mse
    train_mse = train_mse / len(train_data)
    train_mse_wld = train_mse_wld / len(train_data)
    print('mean-square error of 1 sample: %f, #train_data = %d' %
          (train_mse, len(train_data)))
    print('average estimation error in world coordinate system: %f (mm)' %
          (train_mse_wld))

    torch.save(netR.state_dict(), '%s/netR_%d.pth' % (save_dir, epoch))
    torch.save(optimizer.state_dict(), '%s/optimizer_%d.pth' %
               (save_dir, epoch))
