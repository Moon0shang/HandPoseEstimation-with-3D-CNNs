import datetime
from time import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler


from Model import DenseNet
from pre.MSRA_pre import *


AUGMENTED = False
"""
conv3d
relu
BN
dropout
FC: full connected
regression
"""


def init_parser():
    parser = argparse.ArgumentParser(description='TSDF Fusion')
    parser.add_argument('-data', default=DATA_DIR, type=str, metavar='DIR',
                        help='path to dataset(default: {})'.format(DATA_DIR))

    parser.add_argument('-e', '--epochs',  default=EPOCH_COUNT, type=int,
                        help='number of total epochs to run (default: {})'.format(EPOCH_COUNT))

    parser.add_argument('-s', '--start-epoch',  default=0, type=int,
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('-b', '--batch-size', default=BATCH_SIZE, type=int,
                        help='mini-batch size (default: {})'.format(BATCH_SIZE))

    parser.add_argument('-lr', '--learning-rate', default=LEARNING_RATE, type=float,
                        metavar='LR', help='initial learning rate (default: {})'.format(LEARNING_RATE))

    parser.add_argument('-m', '--momentum', default=MOMENTUM, type=float, metavar='M',
                        help='momentum (default: {})'.format(MOMENTUM))

    parser.add_argument('-wd', '--weight-decay', default=WEIGHT_DECAY, type=float,
                        metavar='W', help='weight decay (default: {})'.format(WEIGHT_DECAY))

    parser.add_argument('-p', '--print-freq', default=PRINT_FREQ, type=int,
                        help='print frequency (default: {})'.format(PRINT_FREQ))

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    global args
    args = parser.parse_args()


def main(Model, full=False):
    start_time = time()
    init_parser()
    net = Model()
    net.cuda()
    criterion = nn.MSELoss().cuda()
    best_acc = 0
    optimizer = optim.SGD(net.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    dataset = Read_MSRA()
    train_idx, valid_idx = dataset.get_indices()
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(valid_idx)

    if full:
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=WORKER)
        test_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=args.batch_size,
                                                  num_workers=WORKER)

    else:
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=args.batch_size, sampler=train_sampler,
                                                   num_workers=WORKER)

        test_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=args.batch_size, sampler=test_sampler,
                                                  num_workers=WORKER)
    for epoch in range(args.epochs):
        epoch_start_time = time()
        adjust_learning_rate(optimizer, epoch+1)

        acc = train_step(train_loader, net, criterion, optimizer, epoch+1)
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        print('Epoch: [{0}/{1}]  Time [{2}/{3}]'.format(
            epoch + 1, args.epochs,
            datetime.timedelta(seconds=(time() - epoch_start_time)),
            datetime.timedelta(seconds=(time() - start_time))))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': '3dDNN',
            'state_dict': net.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)
    print('Finished Training')


def train_step(train_loader, model, criterion, optimizer, epoch):

    # switch to train mode
    model.train()

    end = time()
    for i, s in enumerate(train_loader):
        if len(s) == 3:
            tsdf, target, max_l = s
        else:
            tsdf, target, max_l, mid_p = s
            mid_p = mid_p.unsqueeze(1)
        max_l = max_l.unsqueeze(1)
        # measure data loading time
        data_time.update(time() - end)
        if AUGMENTED:
            tsdf, angles = tsdf
            input_sensor_var = torch.autograd.Variable(angles.cuda())
        batch_size = tsdf.size(0)
        tsdf = tsdf[:, 2, :, :, :].unsqueeze(1).cuda()

        # normalize target to [0,1]
        n_target = (target.view(batch_size, -1, 3) -
                    mid_p).view(batch_size, -1) / max_l + 0.5
        n_target[n_target < 0] = 0
        n_target[n_target >= 1] = 1

        input_var = torch.autograd.Variable(tsdf)
        target_var = torch.autograd.Variable(n_target.cuda())

        # compute output
        if type(model) is HandSensorNet:
            output = model((input_var, input_sensor_var))
        elif type(model) is HandNet:
            output = model(input_var)
        elif type(model) is SensorNet:
            output = model(input_sensor_var)

        # record loss
        loss = criterion(output, target_var)
        losses.update(loss.data[0], batch_size)

        # measure accuracy
        # unnormalize output to original space
        output = ((output.data.cpu() - 0.5) *
                  max_l).view(batch_size, -1, 3) + mid_p
        output = output.view(batch_size, -1)
        err_t = accuracy_error_thresh_portion_batch(output, target)
        errors.update(err_t, batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time() - end)
        end = time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'acc_in_t {err_t.val:.3f} ({err_t.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, err_t=errors))
    return errors.avg


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.3 every 5 epochs"""
    lr = args.learning_rate * (0.3 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
