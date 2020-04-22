import argparse
import csv
import os
import shutil
import time
from timer import Timer
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import scipy
from scipy.stats import mode

# import model_list
# from model_list import alexnet
# import util
import warnings

import numpy as np
import cv2
from png2dataset import FastMCD_MultiObject

# set the seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)

import sys
import gc

global args, best_prec1

best_prec1 = 0


def test_multi(loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # bin_op.binarization()

    frame_timer = Timer(desc='Frame', printflag=False)

    for i, data_label_list in enumerate(loader):
        frame_timer.start_time()
        for data_label in data_label_list:
            data, target = data_label
            # target = target.cuda(async=True)
            with torch.no_grad():
                data_var = torch.autograd.Variable(data)
                target_var = torch.autograd.Variable(target)
            
            cv_data = data.data.numpy().squeeze().copy()
            cv_data = np.swapaxes(cv_data, 0, 2)
            # rescale data to 0-255
            cv_data -= np.min(cv_data)
            cv_data /= np.max(cv_data)

            cv_data = np.array(255 * cv_data, dtype='uint8')
            cv2.imshow('input data', cv_data)
            cv2.waitKey(20)

            # compute output
            output = None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                output = model(data_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # RESNET
            # dim: 1, 21, 520, 520
            # scores = torch.nn.functional.softmax(output['out'], dim=1).argmax(dim=1)
            # pred = mode(scores.data.numpy())

            # correct += pred == target.data.numpy()
            
            # correct += pred.eq(target.data.view_as(pred)).sum()

            losses.update(loss.data.item(), data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))
        frame_timer.end_time()
    # bin_op.restore()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    frame_times = frame_timer.get_saved_times()
    preprocess_times = loader.dataset.preprocess_timer.get_saved_times()
    with open('frame_times.csv', 'w', newline='') as myfile:  
        wr = csv.writer(myfile)
        for i in range(len(frame_times)):
            wr.writerow([frame_times[i], preprocess_times[i]])

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    print('Pred vs. Actual:', pred.data, target.data)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Pretrained Model Evaluator with fastMCD region proposal')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                        help='model architecture (default: alexnet (ImageNet), resnet (COCO))')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        default=False, help='use pre-trained model')
                        
    parser.add_argument('--fastMCD', action='store', default='../../fastMCD/test/car_results/',
                        help='whether to use multi object fastMCD detection')
    parser.add_argument('--label', action='store', default='car',
                        help='label to use for all image frames')
    args = parser.parse_args()

    model = None
    im_size = 256
    if args.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        im_size = 256
    elif args.arch == 'resnet':
        model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=21, aux_loss=None)
        im_size = 520

    criterion = nn.CrossEntropyLoss()

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            c = float(m.weight.data[0].nelement())
            m.weight.data = m.weight.data.normal_(0, 2.0 / c)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data = m.weight.data.zero_().add(1.0)
            m.bias.data = m.bias.data.zero_()

    # torchvision.set_image_backend('accimage')

    label = 0
    if args.label == 'car':
        label = 475     # car mirror
        # label = 479   # car wheel
    elif args.label == 'cheetah':
        label = 293

    # Data Loader
    proj_loader = torch.utils.data.DataLoader(
        FastMCD_MultiObject(args.fastMCD, lazylabel=label, im_size=im_size)
    )

    print(model)

    if args.fastMCD:
        test_multi(proj_loader, model, criterion)
        exit()


# Car preditions:
# [475],
# [785],
# [726],
# [739],
# [103]]

# Cheetah predictions
# [475],
# [785],
# [726],
# [739],
# [103]]