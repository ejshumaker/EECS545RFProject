import cv2
import imutils
import numpy as np
import os
import sys
from PIL import Image
from .timer import Timer
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .MCDWrapper import *
from .multiBlobDetector import multiObjectFrame


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


class ImageDataset_python(Dataset):
    def __init__(self, fname, thresh=150):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cap = cv2.VideoCapture(fname)
        self.mcd = MCDWrapper()
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
        self.transform = transforms.ToTensor()

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

        frame = self.transform(frame)

        return frame, 3


class ImageDataset_multi(Dataset):
    def __init__(self, root_dir, lazylabel=3, rotate=True):
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
        self.rotate = rotate

        self.preprocess_timer = Timer(desc='Preprocess', printflag=False)
        # Standard normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
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
        # Create a list of tuples (image, label, and bounding box) for each bounding box in each frame
        data_list = []

        # Load Image
        img_name = os.path.join(self.root_dir, self.image_files[2 * idx + 1])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load Mask
        mask_name = os.path.join(self.root_dir, self.image_files[2 * idx])
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        rects = multiObjectFrame(mask, blobSize=0.05, slack=0.01)

        for rect in rects:
            x, y, w, h = rect

            # grab a buffer region around each bounding box
            # X direction
            buffer = 12
            min_val = 0
            max_val = image.shape[1]
            x_new = np.clip(x - buffer, min_val, max_val)
            moved = x - x_new
            w_new = np.clip(w + buffer + moved, min_val, max_val)

            # Y direction
            min_val = 0
            max_val = image.shape[0]
            y_new = np.clip(y - buffer, min_val, max_val)
            moved = y - y_new
            h_new = np.clip(h + buffer + moved, min_val, max_val)

            x = x_new
            w = w_new
            y = y_new
            h = h_new

            roi = image[y:y + h, x:x + w, :].copy()
            # cv2.imshow('mask', mask)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0))
            # cv2.imshow('image w/ roi', image)
            # cv2.imshow('roi', roi)
            # cv2.waitKey(20)

            roi = cv2.resize(roi, (32, 32))
            if self.rotate:
                roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
            roi = self.transforms(roi)
            
            # label everything as a cat (3)
            label = 0
            if self.lazylabel:
                label = self.lazylabel
            data_list.append((roi, label, (x_new, y_new, x_new + w_new, y_new + h_new)))

        self.preprocess_timer.end_time()
        return data_list
