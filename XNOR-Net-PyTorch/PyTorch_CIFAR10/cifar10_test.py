import os, shutil
import sys
import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer

import torch
import cv2
import csv
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from util545 import png2dataset
from util545.timer import Timer
from cifar10_module import CIFAR10_Module

args = None
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def main(hparams):
    # torch.cuda.set_device(hparams.gpu)
    model = CIFAR10_Module(hparams)
    # trainer = Trainer(gpus=[hparams.gpu], default_save_path=os.path.join(os.getcwd(), 'test_temp'))
    trainer = Trainer(default_save_path=os.path.join(os.getcwd(), 'test_temp'))
    trainer.test(model)
    shutil.rmtree(os.path.join(os.getcwd(), 'test_temp'))


def test_multi(model, loader):
    model.eval()
    correct = 0

    # Create timer for frames
    frame_timer = Timer(desc='Frame', printflag=False)

    file_name = args.data.split("/")[-1]
    if file_name == '':
        file_name = args.data.split("/")[-2]
    resultsFile = open(file_name + '_BOUNDING_BOX_' + args.classifier + '_normal.txt', 'w')

    timed = False

    num_evals = 0
    acc = 0
    for frame, data_list in enumerate(loader):
        frame_timer.start_time()
        if not timed:
            resultsFile.write("frame " + str(frame) + ':\n' + 'Objects:\n\n')
        for data in data_list:
            with torch.no_grad():
                data, target, bounding_box = data
                
                # data, target = Variable(data), Variable(target)
                output = None

                output = model.model(data)
                # test_loss += criterion(output, target).data.item()
                num_evals += 1

                pred = output.data.max(1, keepdim=True)[1]

                if not timed:
                    x1, y1, x2, y2 = bounding_box
                    t2s = lambda x: str(x.numpy()[0])
                
                    # Write bounding box and prediction to file
                    resultsFile.write(classes[pred] + ':\n')
                    resultsFile.write('Bounding Box:' + t2s(x1) + ',' + t2s(y1) + ',' + t2s(x2) + ',' + t2s(y2) + '\n')

                    # # # DEBUG: Display Input Image and print network output
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

                if target == 1:
                    correct += (pred == 1).numpy().sum()
                    correct += (pred == 9).numpy().sum()
                else:
                    correct += pred.eq(target.data.view_as(pred)).sum()
                acc = 100. * float(correct) / num_evals
                # print(output)
                # print('Running Acc:', acc)
        frame_timer.end_time()
    
    frame_times = frame_timer.get_saved_times()
    preprocess_times = loader.dataset.preprocess_timer.get_saved_times()
    if timed:
        with open('frame_times.csv', 'w') as myfile:
            wr = csv.writer(myfile)
            for i in range(len(frame_times)):
                wr.writerow([frame_times[i], preprocess_times[i]])

    return acc


def fastMCDtest(hparams):
    model = CIFAR10_Module(hparams)

    loader = torch.utils.data.DataLoader(png2dataset.ImageDataset_multi(args.data, lazylabel=int(args.label), rotate=False))

    print('Test Accuracy:', test_multi(model, loader))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--data_dir', type=str, default='../CIFAR_10/data')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--reduce_lr_per', type=int, default=50)
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'AdamW'])
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--fastMCD', action='store_true', default=False)
    parser.add_argument('--data', action='store', default='../../fastMCD/test/car_results')
    parser.add_argument('--label', action='store', default='1')
    args = parser.parse_args()

    if args.fastMCD:
        fastMCDtest(args)
        exit(0)

    main(args)