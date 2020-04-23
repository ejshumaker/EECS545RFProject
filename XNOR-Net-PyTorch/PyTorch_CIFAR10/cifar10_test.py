import os, shutil
import sys
import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer

import torch
import cv2
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from util545 import png2dataset
from util545.timer import Timer
from cifar10_module import CIFAR10_Module


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
    # frame_timer = Timer(desc='Frame', printflag=False)

    num_evals = 0
    acc = 0
    for data_label_list in loader:
        # frame_timer.start_time()
        for data_label in data_label_list:
            with torch.no_grad():
                data, target = data_label
                
                # data, target = Variable(data), Variable(target)
                output = None

                output = model.model(data)
                # test_loss += criterion(output, target).data.item()
                num_evals += 1

                pred = output.data.max(1, keepdim=True)[1]

                if target == 1:
                    correct += (pred == 1).numpy().sum()
                    correct += (pred == 9).numpy().sum()
                else:
                    correct += pred.eq(target.data.view_as(pred)).sum()
                acc = 100. * float(correct) / num_evals
                print(output)
                print('Running Acc:', acc)
        # frame_timer.end_time()

    return acc


def fastMCDtest(hparams):
    model = CIFAR10_Module(hparams)

    loader = torch.utils.data.DataLoader(png2dataset.ImageDataset_multi(args.data, lazylabel=int(args.label)))

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