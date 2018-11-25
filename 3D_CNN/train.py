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

subject_names = sorted(os.listdir('./result/'))[:9]


def init_parser():
    parser = argparse.ArgumentParser(description='3D Hand Pose Estimation')
    "write my own ones"
    parser.add_argument('--threshold', type=int, default=20,
                        help='threshold fro calculate proportion')
    parser.add_argument('--batchSize', type=int, default=16,
                        help='input batch size')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers')
    parser.add_argument('--nepoch', type=int, default=5,
                        help='number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.005,
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
    tr_s = []
    tr_s_w = []
    te_s = []
    te_s_e = []

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
    train_data = MSRA_Dataset(root_path='./result',  train=True)
    train_dataloder = DataLoader(
        train_data, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    test_data = MSRA_Dataset(root_path='./result',  train=False)
    test_dataloder = DataLoader(
        test_data, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))
    print('#Train data:', len(train_data), '#Test data:', len(test_data))
    # print(opt)

    # define model,loss and optimizer
    net = DenseNet()
    if opt.model != '':
        net.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))
    'hardware problem'
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
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # some extra datas
    if opt.size == 'full':
        extra_data = [train_data.PCA_mean, train_data.PCA_coeff]
    else:
        extra_data = None
    train_len = len(train_data)
    test_len = len(test_data)

    for epoch in range(opt.nepoch):
        # adjust learning rate
        scheduler.step(epoch)
        # adjest_lr(optimizer, epoch)
        print('======>>>>> Online epoch: #%d/%d, lr=%f, Test: %s <<<<<======' %
              (epoch, opt.nepoch, scheduler.get_lr()[0], subject_names[opt.test_index]))

        # train step
        train_mse, train_mse_wld, timer = train(
            net, extra_data, train_dataloder, criterion, optimizer)

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
        torch.save(net.state_dict(), '%s/netR_%d.pth' % (save_dir, epoch))
        torch.save(optimizer.state_dict(), '%s/optimizer_%d.pth' %
                   (save_dir, epoch))
        # evaluation step
        test_mse, test_wld_err, timer = evaluate(
            net, extra_data, train_dataloder, criterion, optimizer)

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

        # log
        logging.info('Epoch#%d: train error=%e, train wld error = %f mm, test error=%e, test wld error = %f mm, lr = %f' % (
            epoch, train_mse, train_mse_wld, test_mse, test_wld_err, scheduler.get_lr()[0]))
        tr_s.append(train_mse)
        tr_s_w.append(train_mse_wld)
        te_s.append(test_mse)
        te_s_e.append(test_wld_err)
        np.savez('./rt.npz', tr_s=tr_s, tr_s_w=tr_s_w,
                 te_s=te_s, te_s_e=te_s_e)


def train(net, extra_data, train_dataloder, criterion, optimizer):

    torch.cuda.synchronize()
    net.train()
    train_mse = 0.0
    train_mse_wld = 0.0
    start_time = time()

    for i, data in enumerate(tqdm(train_dataloder, 0)):
        torch.cuda.synchronize()
        # load datas
        # joint = ground_truth in dataset file, cause the spell is too long
        if opt.size == 'full':
            [PCA_mean, PCA_coeff] = extra_data
            tsdf, joint, max_l, mid_p, joint_pca = data
            b_size = len(tsdf)
            tsdf = Variable(tsdf, requires_grad=False).cuda()
            joint_pca = Variable(joint_pca, requires_grad=False)
            # 3.1.2 compute output
            optimizer.zero_grad()
            estimation = net(tsdf)
            loss = criterion(estimation, joint_pca)*opt.PCA_SZ

            # 3.1.3 compute gradient and do SGD step
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            # 3.1.4 update training error
            train_mse = train_mse + loss.data[0] * b_size

            # 3.1.5 compute error in world cs
            ml = max_l.unsqueeze(1)
            mp = mid_p.unsqueeze(1)
            outputs_nor = PCA_mean.expand(
                estimation.data.size(0), PCA_mean.size(1))
            # addmm: out = outputs_nor+estimation.data*train_data.PCA_coeff
            outputs_nor = torch.addmm(
                outputs_nor, estimation.data.cpu(), PCA_coeff)
            output = ((outputs_nor - 0.5) * ml).view(b_size, -1, 3) + mp
            proportion, err_mean = cal_out(output, joint, b_size)
        elif opt.size == 'small':
            tsdf, joint, max_l, mid_p = data
            b_size = len(tsdf)
            tsdf = Variable(tsdf, requires_grad=False).cuda()

            # joint transfer and normalization
            joint1 = joint.view(b_size, -1, 3)
            joint_nor = torch.FloatTensor(joint1.size())
            max_l, mid_p = max_l.cuda(), mid_p.cuda()
            for b in range(b_size):
                joint_nor[b] = (joint1[b] - mid_p[b]) / max_l[b] + 0.5

            joint_nor[joint_nor < 0] = 0
            joint_nor[joint_nor > 1] = 1
            joint_nor = joint_nor.view(b_size, -1)
            joint_nor = Variable(joint_nor, requires_grad=False).cuda()

            # compute output
            optimizer.zero_grad()
            estimation = net(tsdf)
            loss = criterion(estimation, joint_nor) * opt.PCA_SZ

            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            # update training error
            train_mse = train_mse + loss.item() * b_size

            # calculate error threshod
            ml = max_l.unsqueeze(1)
            mp = mid_p.unsqueeze(1)
            output = ((estimation.data - 0.5)
                      * ml).view(b_size, -1, 3) + mp
            proportion, err_mean = cal_out(output, joint, b_size)
        else:
            print('wrong opt.size which cause wrong data')
            break

        # infromation output
        if i % 10 == 0:
            print('Train:[%d/%d]\t' % (i, len(train_dataloder)),
                  'Loss:%.4f\t' % loss.item(),
                  'Proportion:%.3f' % proportion)

    torch.cuda.synchronize()
    end_time = time()
    timer = end_time - start_time

    return train_mse, err_mean, timer


def evaluate(net, extra_data, test_dataloder, criterion, optimizer):

    torch.cuda.synchronize()
    net.eval()
    test_mse = 0.0
    test_wld_err = 0.0
    start_time = time()

    for i, data in enumerate(test_dataloder):
        torch.cuda.synchronize()

        if opt.size == 'full':
            [PCA_mean, PCA_coeff] = extra_data
            tsdf, joint, max_l, mid_p, joint_pca = data
            b_size = len(tsdf)
            tsdf = Variable(tsdf, requires_grad=False).cuda()
            joint_pca = Variable(joint_pca, requires_grad=False)
            # 3.1.2 compute output
            optimizer.zero_grad()
            estimation = net(tsdf)
            loss = criterion(estimation, joint_pca)*opt.PCA_SZ

            # 3.1.3 compute gradient and do SGD step
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            # 3.1.4 update training error
            train_mse = train_mse + loss.data[0] * b_size

            # 3.1.5 compute error in world cs
            ml = max_l.unsqueeze(1)
            mp = mid_p.unsqueeze(1)
            outputs_nor = PCA_mean.expand(
                estimation.data.size(0), PCA_mean.size(1))
            # addmm: out = outputs_nor+estimation.data*train_data.PCA_coeff
            outputs_nor = torch.addmm(
                outputs_nor, estimation.data.cpu(), PCA_coeff)
            output = ((outputs_nor - 0.5) * ml).view(b_size, -1, 3) + mp
            proportion, err_mean = cal_out(output, joint, b_size)
        elif opt.size == 'small':
            tsdf, joint, max_l, mid_p = data
            b_size = len(tsdf)
            tsdf = Variable(tsdf, requires_grad=False).cuda()

            # joint transfer and normalization
            joint1 = joint.view(b_size, -1, 3)
            joint_nor = torch.FloatTensor(joint1.size())
            max_l, mid_p = max_l.cuda(), mid_p.cuda()
            for b in range(b_size):
                joint_nor[b] = (joint1[b] - mid_p[b]) / max_l[b] + 0.5

            joint_nor[joint_nor < 0] = 0
            joint_nor[joint_nor > 1] = 1
            joint_nor = joint_nor.view(b_size, -1)
            joint_nor = Variable(joint_nor, requires_grad=False).cuda()

            # compute output
            optimizer.zero_grad()
            estimation = net(tsdf)
            loss = criterion(estimation, joint_nor) * opt.PCA_SZ

            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            # update training error
            train_mse = train_mse + loss.item() * b_size

            # calculate error threshod
            ml = max_l.unsqueeze(1)
            mp = mid_p.unsqueeze(1)
            output = ((estimation.data - 0.5)
                      * ml).view(b_size, -1, 3) + mp
            proportion, err_mean = cal_out(output, joint, b_size)
        else:
            print('wrong opt.size which cause wrong data')
            break

        # infromation output
        if i % 10 == 0:
            print('Train:[%d/%d]\t' % (i, len(test_dataloder)),
                  'Loss:%.4f\t' % loss.item(),
                  'Proportion:%.3f' % proportion)

    torch.cuda.synchronize()
    end_time = time()
    timer = end_time - start_time

    return test_mse, test_wld_err, timer


def cal_out(output, joint, b_size):

    output = output.view(b_size, -1)
    diff = torch.abs(output-joint).view(b_size, -1, 3)
    sqr_sum = torch.sum(torch.pow(diff, 2), 2)
    sqrt_sum = torch.sqrt(sqr_sum)
    # calculate propotion
    out = torch.zeros(sqrt_sum.size())
    t = opt.threshold
    out[sqr_sum < t] = 1
    good = torch.sum(out) / (out.size(1) * b_size)
    # error in world corrdinate system
    err_mean = torch.mean(sqrt_sum, 1).view(-1, 1)
    err_mean = torch.sum(err_mean)

    return good * 100, err_mean


def visualize():
    pass


if __name__ == "__main__":
    main()
