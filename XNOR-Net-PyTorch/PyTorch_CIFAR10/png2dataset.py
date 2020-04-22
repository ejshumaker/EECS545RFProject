import cv2
import imutils
import numpy as np
import os
import sys
from PIL import Image
from timer import Timer
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from multiBlobDetector import multiObjectFrame

class ImageDataset(Dataset):
    def __init__(self, root_dir, thresh=150):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_files = []
        for filename in os.listdir(root_dir):
            if filename.endswith(".png"):
                self.image_files.append(filename)
        self.image_files = sorted(self.image_files)
        self.root_dir = root_dir
        self.thresh = thresh
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files) // 2

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load Image
        img_name = os.path.join(self.root_dir, self.image_files[2 * idx + 1])
        image = cv2.imread(img_name)

        # Load Mask
        mask_name = os.path.join(self.root_dir, self.image_files[2 * idx])
        mask = cv2.imread(mask_name)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        x, y, w, h = cv2.boundingRect(mask)

        crop_image = image[y:y + h, x:x + w, :]
        if crop_image.size != 0:
            image = cv2.resize(crop_image, (32, 32))
        else:
            image = cv2.resize(image, (32, 32))
        cv2.imshow('image', image)
        cv2.waitKey(10)

        image = self.transform(image.astype('float32'))

        return image, 3


class ImageDataset_multi(Dataset):
    '''
    TODO: need to be clever about how to properly load images.
    One image may have multiple objects to classify, so we will
    have to circumvent the __getitem__ process to return multiple items,
    or not use a dataloader and instead simply call the forward pass on the
    network ourselves.
    '''
    def __init__(self, root_dir, lazylabel=3):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_files = []
        for filename in os.listdir(root_dir):
            if filename.endswith(".png"):
                self.image_files.append(filename)
        self.image_files = sorted(self.image_files)
        self.root_dir = root_dir

        # If no ground truth for each frame, use lazy label
        self.lazylabel = lazylabel

        self.preprocess_timer = Timer(desc='Preprocess', printflag=False)
        # Standard normalization
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self):
        return len(self.image_files) // 2

    def __getitem__(self, idx):
        self.preprocess_timer.start_time()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_label_list = []

        # Load Image
        img_name = os.path.join(self.root_dir, self.image_files[2 * idx + 1])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load Mask
        mask_name = os.path.join(self.root_dir, self.image_files[2 * idx])
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        rects = multiObjectFrame(mask, blobSize=0.07, slack=0.02)

        for rect in rects:
            x, y, w, h = rect

            # grab a buffer region of 4 pixels to left, right, up, and down for each bounding box
            # X direction
            min_val = 0
            max_val = image.shape[1]
            x_new = np.clip(x - 4, min_val, max_val)
            moved = x - x_new
            w_new = np.clip(w + 4 + moved, min_val, max_val)

            # Y direction
            min_val = 0
            max_val = image.shape[0]
            y_new = np.clip(y - 4, min_val, max_val)
            moved = y - y_new
            h_new = np.clip(h + 4 + moved, min_val, max_val)

            x = x_new
            w = w_new
            y = y_new
            h = h_new

            roi = image[y:y + h, x:x + w, :].copy()
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0))
            # cv2.imshow('image w/ roi', image)
            # cv2.imshow('mask', mask)

            roi = cv2.resize(roi, (32, 32))
            roi = roi#.astype('float32')
            roi = self.transforms(roi)
            
            # label everything as a cat (3)
            label = 0
            if self.lazylabel:
                label = self.lazylabel
            data_label_list.append((roi, label))

        self.preprocess_timer.end_time()
        return data_label_list
