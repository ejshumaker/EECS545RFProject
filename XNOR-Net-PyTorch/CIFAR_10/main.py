from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cifar_data
import csv
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import util
import warnings

from models import nin, vgg, mobilenetv2, resnet_20, resnet_34
from torch.autograd import Variable

import numpy as np
import cv2
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from util545 import png2dataset
from util545.timer import Timer


# Global Variables
args = None
model = None
best_acc = None
model_type = None
# CIFAR-10 Classes
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def save_state(model, best_acc):
    print('Saving model')
    new_state_dict = {}
    for key in model.state_dict().keys():
        new_state_dict[key.replace('module.', '')] = model.state_dict()[key]

    state = {
        'best_acc': best_acc,
        'state_dict': new_state_dict
    }
    torch.save(state, 'models/' + args.arch + '_' + model_type + '.pth.tar')


def train(epoch, loader):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        # process the weights including binarization
        bin_op.binarization()

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        # forward pass
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = model(data)
        
        # backward pass
        loss = criterion(output, target)
        loss.backward()

        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()

        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return


def test(loader):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    bin_op.binarization()
    for data, target in loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        if args.cuda:
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        else:
            correct += pred.eq(target.data.view_as(pred)).sum()
    bin_op.restore()
    acc = 100. * float(correct) / len(loader.dataset)

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)
    
    test_loss /= len(loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(loader.dataset), acc))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    return


def test_multi(loader):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    bin_op.binarization()

    confusion = np.zeros(10)

    # Create timer for frames
    frame_timer = Timer(desc='Frame', printflag=False)
    timed = args.timed

    # Create a file to write frame results to:
    file_name = args.multi_fastMCD.split("/")[-1]
    if file_name == '':
        file_name = args.multi_fastMCD.split("/")[-2]
    resultsFile = open(file_name + '_BOUNDING_BOX_' + args.arch + '_' + model_type + '.txt', 'w')

    num_evals = 0
    acc = 0
    for frame, data_list in enumerate(loader):
        frame_timer.start_time()
        if not timed:
            resultsFile.write("frame " + str(frame) + ':\n' + 'Objects:\n\n')
        for data_label in data_list:
            with torch.no_grad():
                data, target, bounding_box = data_label
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                # data, target = Variable(data), Variable(target)
                output = None
                # Autograd Issue is in model
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    output = model(data)
                test_loss += criterion(output, target).data.item()
                num_evals += 1

                pred = output.data.max(1, keepdim=True)[1]

                if not timed:
                    x1, y1, x2, y2 = bounding_box
                    t2s = lambda x: str(x.numpy()[0])
                    
                    # Write bounding box and prediction to file
                    resultsFile.write(classes[pred] + ':\n')
                    resultsFile.write('Bounding Box:' + t2s(x1) + ',' + t2s(y1) + ',' + t2s(x2) + ',' + t2s(y2) + '\n')

                confusion[pred] += 1

                # DEBUG: Display Input Image and print network output
                # cv_data = data.data.numpy().squeeze().copy()
                # cv_data = np.swapaxes(cv_data, 0, 2)
                # # rescale data to 0-255
                # cv_data -= np.min(cv_data)
                # cv_data /= np.max(cv_data)
                # cv_data = np.array(255 * cv_data, dtype='uint8')
                # cv2.imshow('input data', cv_data)
                # cv2.waitKey(50)
                # print(output)
                # print('Pred vs. Target:', pred, target.data)

                if args.cuda:
                    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
                else:
                    # Treat cars and trucks the same
                    if target == 1:
                        correct += (pred == 1).numpy().sum()
                        correct += (pred == 9).numpy().sum()
                    else:
                        correct += pred.eq(target.data.view_as(pred)).sum()

                acc = 100. * float(correct) / num_evals
                # print('Running Accuracy', acc)
        frame_timer.end_time()
    bin_op.restore()
    resultsFile.close()

    # if acc > best_acc:
    #     best_acc = acc
    #     save_state(model, best_acc)

    test_loss /= num_evals
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, num_evals, acc))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))

    frame_times = frame_timer.get_saved_times()
    preprocess_times = loader.dataset.preprocess_timer.get_saved_times()
    if timed:
        with open('frame_times.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            for i in range(len(frame_times)):
                wr.writerow([frame_times[i], preprocess_times[i]])

    return confusion


def adjust_learning_rate(optimizer, epoch):
    update_list = [120, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return


if __name__ == '__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
                        help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='./data/',
                        help='dataset path')
    parser.add_argument('--arch', action='store', default='nin',
                        help='the architecture for the network: [nin, vgg, resnet20, or resnet34]')
    parser.add_argument('--lr', action='store', default='0.01',
                        help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,
                        help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model')

    parser.add_argument('--fastMCD', action='store', default=None,
                        help='whether to run on fastMCD data')
    parser.add_argument('--python_fastMCD', action='store', default=None,
                        help='whether to use python fastMCD on data or not')
    parser.add_argument('--multi_fastMCD', action='store', default=None,
                        help='whether to use multi object fastMCD detection')
    parser.add_argument('--label', action='store', default=None,
                        help='label to use for all image frames')
    parser.add_argument('--bwn', action='store_true', default=False,
                        help='Use binary weight network only (do not binarize inputs)')
    parser.add_argument('--timed', action='store_true', default=False,
                        help='whether or not to time this run')
    args = parser.parse_args()
    print('==> Options:', args)

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    args.cuda = not args.cpu and torch.cuda.is_available()

    if not args.pretrained:
        trainset = cifar_data.original_dataset(root=args.data, train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True) #, num_workers=2)

        testset = cifar_data.original_dataset(root=args.data, train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False) #, num_workers=1)

    # define classes

    # 545 Project Data
    proj_loader = None
    if args.fastMCD:
        proj_loader = torch.utils.data.DataLoader(
            png2dataset.ImageDataset(args.fastMCD, thresh=160), shuffle=True)
    if args.python_fastMCD:
        proj_loader = torch.utils.data.DataLoader(
            png2dataset.ImageDataset_python(args.python_fastMCD, thresh=160), shuffle=True)
    if args.multi_fastMCD:
        proj_loader = torch.utils.data.DataLoader(
            png2dataset.ImageDataset_multi(args.multi_fastMCD, lazylabel=int(args.label)), shuffle=False)

    # define the model
    model_type = 'XNOR'
    if args.bwn:
        model_type = 'BWN'
    print('==> building model', args.arch, '...')
    if args.arch == 'nin':
        model = nin.Net(init_weights=not args.pretrained, bwn=args.bwn)
    elif args.arch == 'vgg':
        model = vgg.VGG13(init_weights=not args.pretrained, bwn=args.bwn)
    elif args.arch == 'resnet20':
        model = resnet_20.ResNet20(resnet_20.BasicBlock, [3, 3, 3], bwn=args.bwn)
    elif args.arch == 'resnet34':
        model = resnet_34.ResNet34(resnet_34.BasicBlock, [3, 4, 6, 3], bwn=args.bwn)
    else:
        raise Exception(args.arch + ' is currently not supported')

    # initialize the model if pretrained
    if args.pretrained:
        print('==> Load pretrained model from', args.pretrained, '...')
        if args.cuda:
            pretrained_model = torch.load(args.pretrained)
        else:
            pretrained_model = torch.load(args.pretrained, map_location=torch.device('cpu'))
        best_acc = pretrained_model['best_acc']
        new_state_dict = {}
        for key in pretrained_model['state_dict'].keys():
            new_state_dict[key.replace('module.', '')] = pretrained_model['state_dict'][key]
        model.load_state_dict(new_state_dict)

    if args.cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)

    # define solver and criterion
    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params': [value], 'lr': base_lr, 'weight_decay':0.00001}]

    optimizer = optim.Adam(params, lr=0.10, weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss()

    bin_op = util.BinOp(model)

    # do the evaluation if specified
    if args.multi_fastMCD:
        confusion = test_multi(proj_loader)
        print('Label:', args.label)
        print('Confusion:', confusion)
        exit(0)
    if args.python_fastMCD:
        test(proj_loader)
        exit(0)
    if args.fastMCD:
        test(proj_loader)
        exit(0)
    if args.evaluate:
        test(testloader)
        exit(0)

    # start training
    for epoch in range(1, 320):
        adjust_learning_rate(optimizer, epoch)
        train(epoch, trainloader)
        test(testloader)
