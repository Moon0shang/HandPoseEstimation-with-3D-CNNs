"""
training
"""
import os
import shutil
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
    "write my own ones"
    parser.add_argument('--batchSize', type=int,
                        default=16, help='input batch size')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers')
    parser.add_argument('--nepoch', type=int, default=60,
                        help='number of epochs to train for')
    parser.add_argument('--learning_rate', type=float,
                        default=0.01, help='learning rate at t=0')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (SGD only)')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0005, help='weight decay (SGD only)')
    parser.add_argument('--size', type=str, default='full',
                        help='how many samples do we load: small | full')
    parser.add_argument('--test_index', type=int, default=3,
                        help='test index for cross validation, range: 0~8')
    parser.add_argument('--optimizer', type=str, default='',
                        help='optimizer name for training resume')
    # depends on the dataset's ground truth joint numbers, here is 21*3
    parser.add_argument('--PCA_SZ', type=int, default=63,
                        help='number of PCA components')
    # parser.add_argument('--batchSize', type=int,
    #                     default=32, help='input batch size')
    # parser.add_argument('--workers', type=int, default=0,
    #                     help='number of data loading workers')
    # parser.add_argument('--nepoch', type=int, default=60,
    #                     help='number of epochs to train for')
    # parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
    # # CUDA_VISIBLE_DEVICES=0 python train.py
    # parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id')
    # parser.add_argument('--learning_rate', type=float,
    #                     default=0.001, help='learning rate at t=0')
    # parser.add_argument('--momentum', type=float, default=0.9,
    #                     help='momentum (SGD only)')
    # parser.add_argument('--weight_decay', type=float,
    #                     default=0.0005, help='weight decay (SGD only)')
    # parser.add_argument('--learning_rate_decay', type=float,
    #                     default=1e-7, help='learning rate decay')
    # parser.add_argument('--size', type=str, default='full',
    #                     help='how many samples do we load: small | full')
    # parser.add_argument('--JOINT_NUM', type=int, default=21,
    #                     help='number of joints')
    # parser.add_argument('--test_index', type=int, default=3,
    #                     help='test index for cross validation, range: 0~8')
    # parser.add_argument('--save_root_dir', type=str,
    #                     default='results',  help='output folder')
    # parser.add_argument('--model', type=str, default='',
    #                     help='model name for training resume')
    # parser.add_argument('--optimizer', type=str, default='',
    #                     help='optimizer name for training resume')

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
    optimizer = optim.SGD(
        net.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay)

    if opt.optimizer != '':
        optimizer.load_state_dict(torch.load(
            os.path.join(save_dir, opt.optimizer)))
    # auto adjust learning rate, divided by 10 after 50 rpoch
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    best_result = 0
    for epoch in range(opt.nepoch):
        # adjust learning rate
        scheduler.step(epoch)
        print('======>>>>> Online epoch: #%d/%d, lr=%f, Test: %s <<<<<======' %
              (epoch, opt.nepoch, scheduler.get_lr()[0], subject_names[opt.test_index]))
        start_time = time()
        losses, error_avg = train(net, train_dataloder,
                                  criterion, optimizer, epoch)
        end_time = time()
        timer = (end_time-start_time)/len(train_data)
        print('==> time to learn 1 sample = %f (ms)' % timer)
        print('mean-square error of 1 sample: %f, #train_data = %d' %
              (losses, len(train_data)))

        if error_avg > best_result:
            best_result = error_avg
            state = {'epoch': epoch + 1,
                     'state_dict': net.state_dict(),
                     'best': best_result,
                     'optimizer': optimizer.state_dict()}
            torch.save(state, './check_point.pth.tar')


def train(net, train_dataloder, criterion, optimizer, epoch):

    losses = 0.0
    errors = 0.0
    # swich to train mode
    torch.cuda.synchronize()
    net.train()

    for i, data in enumerate(train_dataloder):

        # load input and target
        tsdf, ground_truth, ground_truth_pca, max_l, mid_p = data
        mid_p = mid_p.unsqueeze(1)
        max_l = max_l.unsqueeze(1)
        batch_size = tsdf.size(0)
        # normalize target to [0,1]
        target = (ground_truth.view(batch_size, -1, 3) -
                  mid_p).view(batch_size, -1) / max_l + 0.5
        target[target < 0] = 0
        target[target >= 1] = 1

        tsdf_var = Variable(tsdf)
        target_var = Variable(ground_truth)

        output = net(tsdf_var)

        # record loss
        loss = criterion(output, target_var)
        losses += loss.data[0]

        # measure accuracy
        # unnormalize output to original sapce
        "need to complete"
        output = ((output.data.cuda() - 0.5) *
                  max_l).view(batch_size, -1, 3) + mid_p
        output = output.view(batch_size, -1)
        err_t = accuracy_error_thresh_portion_batch(output, target)
        errors += err_t

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses, errors/batch_size


def evaluate():
    pass


def visualize():
    pass


def save_checkpoint(state, is_best):
    torch.save(state, file_nema)
    if is_best:
        shutil.copyfile(filename, 'best.tar')


def accuracy_error_thresh_portion_batch(output, target, t=30.0):
    batch_size = target.size(0)
    sample_size = target.size(1)
    diff = torch.abs(output-target).view(batch_size, -1, 3)
    sqr_sum = torch.sum(torch.pow(diff, 2), 2)
    out = torch.zeros(sqr_sum.size())
    t = t**2
    out[sqr_sum < t] = 1
    good = torch.sum(out)/(out.size(1)*batch_size)
    return good*100


if __name__ == "__main__":
    main()
