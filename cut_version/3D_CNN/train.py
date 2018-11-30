"""
training and evaluation
"""
import os
import random
import logging
import argparse
import numpy as np
from time import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataset import MSRA_Dataset
from network import DenseNet

# set the GPU devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
file_path = '/media/hp2/96e32a29-1cf9-4ce8-ba6c-706bd9d5f8fa/hp2/xuyuanquan/code/Pose/result'
subject_names = sorted(os.listdir(file_path))[:9]


def init_parser():
    parser = argparse.ArgumentParser(description='3D Hand Pose Estimation')
    """write my own ones"""
    parser.add_argument('--threshold', type=int, default=10,
                        help='threshold fro calculate proportion')
    parser.add_argument('--batchSize', type=int, default=32,
                        help='input batch size')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers')
    parser.add_argument('--nepoch', type=int, default=60,
                        help='number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate at t=0')
    parser.add_argument('--size', type=str, default='small',
                        help='how many samples do we load: small | full')
    parser.add_argument('--test_index', type=int, default=1,
                        help='test index for cross validation, range: 0~8')
    parser.add_argument('--model', type=str, default='',
                        help='model name for training resume')
    parser.add_argument('--optimizer', type=str, default='',
                        help='optimizer name for training resume')
    # depends on the dataset's ground truth joint numbers, here is 21*3
    parser.add_argument('--PCA_SZ', type=int, default=63,
                        help='number of PCA components')
    parser.add_argument('--save_root_dir', type=str,
                        default='./NET',  help='output folder')
    global opt
    opt = parser.parse_args()


def main():

    init_parser()
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    save_dir = os.path.join(opt.save_root_dir, subject_names[opt.test_index])

    try:
        os.mkdir(opt.save_root_dir)
        os.mkdir(save_dir)
        print('create save dir')
    except:
        print('dir already exist!')

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                        filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
    logging.info('======================================================')

    # load data
    train_data = MSRA_Dataset(root_path=file_path, opt=opt, train=True)
    train_dataloder = DataLoader(
        train_data, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    test_data = MSRA_Dataset(root_path=file_path, opt=opt, train=False)
    test_dataloder = DataLoader(
        test_data, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))
    print('#Train data:', len(train_data), '#Test data:', len(test_data))
    # print(opt)

    # define model,loss and optimizer
    net = DenseNet()
    # if opt.model != '':
    #     net.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))
    '''hardware problem'''
    net.cuda()
    # print(net)

    criterion = nn.MSELoss(size_average=True).cuda()
    optimizer = optim.SGD(
        net.parameters(),
        lr=opt.learning_rate,
        momentum=0.9,
        weight_decay=0.0005)

    if opt.optimizer != '':
        optimizer.load_state_dict(torch.load(
            os.path.join(save_dir, opt.optimizer)))
    # auto adjust learning rate, divided by 10 after 50 rpoch
    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    train_len = len(train_data)
    test_len = len(test_data)

    for epoch in range(opt.nepoch):
        # adjust learning rate
        scheduler.step(epoch)
        # adjest_lr(optimizer, epoch)
        print('======>>>>> Online epoch: #%d/%d, lr=%f, Test: %s <<<<<======' %
              (epoch+1, opt.nepoch, scheduler.get_lr()[0], subject_names[opt.test_index]))

        # train step
        train_mse, train_mse_wld, timer = train(
            net, train_dataloder, criterion, optimizer)

        # time cost
        timer = timer / train_len
        print('==> time to learn 1 sample = %f (ms)' % (timer*1000))

        # print mse
        train_mse = train_mse / train_len
        train_mse_wld = train_mse_wld / train_len
        print('mean-square error of 1 sample: %f, #train_data = %d' %
              (train_mse, train_len))
        print('average estimation error in world coordinate system: %f (mm)' %
              (train_mse_wld))

        # save net
        store = bool(epoch == opt.nepoch-1)
        if store:
            torch.save(net.state_dict(), '%s/net_%d.pth' % (save_dir, epoch))
            torch.save(optimizer.state_dict(), '%s/optimizer_%d.pth' %
                       (save_dir, epoch))
        logging.info('Epoch#%d: train error=%e, train wld error = %f mm,  lr = %f' % (
            epoch+1, train_mse, train_mse_wld, scheduler.get_lr()[0]))

        # evaluation step
        # store = True

        test_mse, test_wld_err, timer = evaluate(
            net, test_dataloder, criterion, optimizer, store)

        # time cost
        timer = timer / test_len
        print('==> time to learn 1 sample = %f (ms)' % (timer*1000))

        # print mse
        test_mse = test_mse / test_len
        print('mean-square error of 1 sample: %f, #test_data = %d' %
              (test_mse, test_len))
        test_wld_err = test_wld_err / test_len
        print('average estimation error in world coordinate system: %f (mm)' %
              (test_wld_err))
        logging.info('Epoch#%d:  test error=%e, test wld error = %f mm, lr = %f' % (
            epoch+1, test_mse, test_wld_err, scheduler.get_lr()[0]))


def train(net, train_dataloder, criterion, optimizer):

    torch.cuda.synchronize()
    net.train()
    train_mse = 0.0
    train_mse_wld = 0.0

    start_time = time()
    for i, data in enumerate(train_dataloder):
        torch.cuda.synchronize()
        # load datas
        # joint = ground_truth in dataset file, cause the spell is too long

        tsdf, joint, max_l, mid_p, joint_nor = data
        b_size = len(tsdf)
        tsdf, joint, joint_nor = tsdf.cuda(), joint.cuda(), joint_nor.cuda()
        max_l, mid_p = max_l.cuda(), mid_p.cuda()
        train_tsdf = Variable(tsdf, requires_grad=False)
        train_joint_nor = Variable(joint_nor, requires_grad=False)

        # compute output
        optimizer.zero_grad()
        train_est = net(train_tsdf)
        train_loss = criterion(train_est, train_joint_nor) * opt.PCA_SZ

        # compute gradient and do SGD step
        train_loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        # update training error
        train_mse = train_mse + train_loss.item() * b_size

        # calculate error threshod
        ml = max_l.unsqueeze(1)
        mp = mid_p.unsqueeze(1)
        output = ((train_est.data - 0.5)
                  * ml).view(b_size, -1, 3) + mp
        proportion, err_mean = cal_out(output, joint, b_size)

        # infromation output
        if i % 20 == 0:
            print('Train:[%4d/%4d]\t' % (i, len(train_dataloder)),
                  'Loss:%.4f\t' % train_loss.item(),
                  'Proportion:%.3f' % proportion)
        train_mse_wld += err_mean

    torch.cuda.synchronize()
    end_time = time()
    timer = end_time - start_time

    return train_mse, train_mse_wld, timer


def evaluate(net, test_dataloder, criterion, optimizer, store):

    torch.cuda.synchronize()
    net.eval()
    test_mse = 0.0
    test_wld_err = 0.0
    if store:
        try:
            os.mkdir('./output')
            os.mkdir('./output/joint')
            os.mkdir('./output/out')
            print('create output files')
        except:
            print('failed create output files!')
    start_time = time()

    for i, data in enumerate(test_dataloder):
        torch.cuda.synchronize()

        tsdf, joint, max_l, mid_p, joint_nor = data
        b_size = len(tsdf)
        tsdf, joint, joint_nor = tsdf.cuda(), joint.cuda(), joint_nor.cuda()
        max_l, mid_p = max_l.cuda(), mid_p.cuda()
        test_tsdf = Variable(tsdf, requires_grad=False)
        test_joint_nor = Variable(joint_nor, requires_grad=False)

        # compute output
        optimizer.zero_grad()
        test_est = net(test_tsdf)
        test_loss = criterion(test_est, test_joint_nor) * opt.PCA_SZ

        torch.cuda.synchronize()

        # update training error
        test_mse = test_mse + test_loss.item() * b_size

        # calculate error threshod
        ml = max_l.unsqueeze(1)
        mp = mid_p.unsqueeze(1)
        output = ((test_est.data - 0.5)
                  * ml).view(b_size, -1, 3) + mp
        proportion, err_mean = cal_out(output, joint, b_size)

        # record datas
        test_wld_err += err_mean
        if store:
            # all outputs
            outs = np.array(output.cpu())
            joints = np.array(joint.cpu())
            np.save('./output/out/%s.npy' % i, outs)
            np.save('./output/joint/%s.npy' % i, joints)
        # infromation output
        if i % 20 == 0:
            print('Test:[%4d/%4d]\t' % (i, len(test_dataloder)),
                  'Loss:%.4f\t' % test_loss.item(),
                  'Proportion:%.3f' % proportion)

    torch.cuda.synchronize()
    end_time = time()
    timer = end_time - start_time

    return test_mse, test_wld_err, timer


def cal_out(output, joint, b_size):

    # calculate mean square errors
    diff = torch.abs(output-joint).view(b_size, -1, 3)
    sqr_sum = torch.sum(torch.pow(diff, 2), 2)
    sqrt_sum = torch.sqrt(sqr_sum)
    # calculate propotion
    out = torch.zeros(sqrt_sum.size())
    t = opt.threshold
    out[sqrt_sum < t] = 1
    good = torch.sum(out) / (out.size(1) * b_size)
    # error in world corrdinate system
    err_mean = torch.mean(sqrt_sum, 1).view(-1, 1)
    err_mean = torch.sum(err_mean)

    return good * 100, err_mean


if __name__ == "__main__":
    main()
