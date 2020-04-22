import os
import torch
import pickle
import numpy
import torchvision.transforms as transforms
import cv2
import numpy as np

class dataset():
    def __init__(self, root=None, train=True):
        self.root = root
        self.train = train
        self.transform = transforms.ToTensor()
        if self.train:
            train_data_path = os.path.join(root, 'train_data')
            train_labels_path = os.path.join(root, 'train_labels')
            self.train_data = numpy.load(open(train_data_path, 'rb'))
            self.train_data = torch.from_numpy(self.train_data.astype('float32'))
            self.train_labels = numpy.load(open(train_labels_path, 'rb')).astype('int')
        else:
            test_data_path = os.path.join(root, 'test_data')
            test_labels_path = os.path.join(root, 'test_labels')
            self.test_data = numpy.load(open(test_data_path, 'rb'))
            self.test_data = torch.from_numpy(self.test_data.astype('float32'))
            self.test_labels = numpy.load(open(test_labels_path, 'rb')).astype('int')

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        
        # 2 notes: images are saved as float32 arrays, with min = ~-3 and max = ~3
        # cv_img = img.data.numpy().copy()
        # print(np.max(cv_img), np.min(cv_img))
        # cv_img = np.swapaxes(cv_img, 0, 2)
        # cv2.imshow('img', cv_img)
        # cv2.imshow('rgb2bgr img', cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1000)
        return img, target


class original_dataset():
    def __init__(self, root=None, train=True):
        self.root = root
        self.train = train
        self.train_data = np.zeros((50000, 3072), dtype='uint8')
        self.train_labels = np.zeros(50000)
        self.test_data = np.zeros((10000, 3072), dtype='uint8')
        self.test_labels = np.zeros(10000)
        data_path = os.path.join(root, 'cifar-10-batches-py')

        # Standard normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        if self.train:
            for i in range(5):
                fname = os.path.join(data_path, 'data_batch_' + str(i + 1))
                data = None
                with open(fname, 'rb') as fo:
                    data = pickle.load(fo, encoding='bytes')
                
                # 10000x3072 numpy array of uint8s, each row is a 32x32 color image. First 1024 entries are the row-major red entries and so on
                self.train_data[i * 10000:(i + 1) * 10000] = data[b'data']
                self.train_labels[i * 10000:(i + 1) * 10000] = data[b'labels']

            # convert image from 10000x3072 arrays to 10000x32x32x3 arrays
            self.train_data = self.train_data.reshape((50000, 32, 32, 3), order='F').astype('float32')
            self.train_labels = self.train_labels.astype('int')
        else:
            fname = os.path.join(data_path, 'test_batch')
            data = None
            with open(fname, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
            
            # 10000x3072 numpy array of uint8s, each row is a 32x32 color image. First 1024 entries are the row-major red entries and so on
            self.test_data = data[b'data']
            self.test_labels[:] = data[b'labels']

            self.test_data = self.test_data.reshape((10000, 32, 32, 3), order='F').astype('float32')
            self.test_labels = self.test_labels.astype('int')

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        
        # 2 notes: images are saved as float32 arrays, with min = ~-3 and max = ~3

        print(img.shape, np.max(img), np.min(img))
        # cv_img = np.swapaxes(img, 0, 2)
        cv2.imshow('img', img.astype('uint8'))
        cv2.imshow('rgb2bgr img', cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1000)

        img = self.transforms(img)

        return img, target
