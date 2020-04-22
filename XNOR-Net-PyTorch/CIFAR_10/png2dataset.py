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

import MCDWrapper
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


class ImageDataset_python(Dataset):
    def __init__(self, fname, thresh=150):
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

        frame = self.transform(frame.astype('float32'))

        return frame, 3


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
        self.transform = transforms.ToTensor()

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

        # Load Mask
        mask_name = os.path.join(self.root_dir, self.image_files[2 * idx])
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        rects = multiObjectFrame(mask, blobSize=0.07, slack=0.02)

        for rect in rects:
            x, y, w, h = rect

            roi = image[y:y + h, x:x + w, :].copy()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0))
            cv2.imshow('image w/ roi', image)
            cv2.imshow('mask', mask)
            if roi.size != 0:
                roi = cv2.resize(roi, (32, 32))
            else:
                roi = cv2.resize(image, (32, 32))
            cv2.imshow('roi', roi)

            # # CIFAR-10 Dataset does some preprocessing. This is hard to do as we must have ALL
            # # image patches available in order to perform the zca whitening. May need to use a different datset
            # # or network.
            # roi = globalContrastNormalization(roi.astype('float32'))
            # roi = roi.reshape(32 * 32 * 3)
            # roi = zca_whiten(roi)
            # roi = roi.reshape((32, 32, 3))

            cv2.imshow('roi GCN ZCA', roi)
            cv2.waitKey(20)
            
            # label everything as a cat (3)
            label = 0
            if self.lazylabel:
                label = self.lazylabel
            data_label_list.append((roi, label))

        self.preprocess_timer.end_time()
        return data_label_list


def globalContrastNormalization(img, s=1, lam=10, eps=1e-5):
    '''Global contrast normalization on an image. Must be of type float32'''
    img_avg = np.mean(img)
    img -= img_avg

    std_dev = np.sqrt(lam + np.mean(img**2))

    img = s * img / max(std_dev, eps)
    return img


def zca_whiten(img):
    '''
    ZCA Whitening of a set of images. Must by float32.
    '''
    from sklearn.decomposition import PCA

    pca = PCA(whiten=True)
    transformed = pca.fit_transform(img)
    pca.whiten = False
    zca = pca.inverse_transform(transformed)
    return zca


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
