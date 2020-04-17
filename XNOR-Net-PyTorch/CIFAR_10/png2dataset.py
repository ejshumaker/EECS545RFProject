import cv2
import imutils
import numpy as np
import os
import sys
from PIL import Image
import torch
from torch.utils.data import Dataset

import MCDWrapper


class ImageDataset(Dataset):
    def __init__(self, root_dir, thresh=150, transform=None):
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
        self.transform = transform

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
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, 3


class ImageDataset_python(Dataset):
    def __init__(self, fname, thresh=150, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cap = cv2.VideoCapture(fname)
        self.mcd = MCDWrapper.MCDWrapper()
        self.len = 0
        gray = None
        isFirst = True
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if isFirst:
                self.mcd.init(gray)
                isFirst = False
            if not ret:
                break
            self.len += 1
        
        self.mask = np.zeros(gray.shape, np.uint8)
        self.cap = cv2.VideoCapture(fname)
        
        self.thresh = thresh
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        ret, frame = self.cap.read()
        if not ret:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.mask = self.mcd.run(gray)
        # frame[mask > 0, 2] = 255
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(10) & 0xFF == ord('q'):
        #     break

        x, y, w, h = cv2.boundingRect(self.mask)

        crop_image = frame[y:y + h, x:x + w, :]
        if crop_image.size != 0:
            frame = cv2.resize(crop_image, (32, 32))
        else:
            frame = cv2.resize(frame, (32, 32))
        cv2.imshow('image', frame)
        cv2.waitKey(10)
        frame = Image.fromarray(frame)

        if self.transform:
            frame = self.transform(frame)

        return frame, 3


class MultiImageDataset(Dataset):
    '''
    TODO: need to be clever about how to properly load images.
    One image may have multiple objects to classify, so we will
    have to circumvent the __getitem__ process to return multiple items,
    or not use a dataloader and instead simply call the forward pass on the
    network ourselves.
    '''
    def __init__(self, root_dir, thresh=150, transform=None):
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
        self.transform = transform

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
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, 3


# Run as standalone to get images of dataset
if __name__ == '__main__':
    if len(sys.argv <= 2):
        print("Please supply a path to image dataset")
        exit(1)
    path = sys.argv[-1]
    # create Dataset
    dataset = ImageDataset(path)
    image = cv2.cvtColor(np.array(dataset.__getitem__(0)[0].convert('RGB')), cv2.COLOR_RGB2BGR)
    cv2.imshow('image', image)
    cv2.waitKey(5000)
