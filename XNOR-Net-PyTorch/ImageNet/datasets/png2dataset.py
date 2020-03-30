import cv2
import imutils
import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class MaskDataset(Dataset):
    """Sample Dataset"""

    def __init__(self, root_dir, transform=None):
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
        self.transform = transform

    def __len__(self):
        return len(self.image_files) // 2

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Take only image masks
        img_name = os.path.join(self.root_dir, self.image_files[2 * idx])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        x, y, w, h = cv2.boundingRect(image)
        crop_image = image[y:y + h, x:x + w]
        if crop_image.size != 0:
            image = cv2.resize(crop_image, (28, 28))
        else:
            image = cv2.resize(image, (28, 28))
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, 5


class ImageDataset(Dataset):
    """Sample Dataset"""

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

        img_name = os.path.join(self.root_dir, self.image_files[2 * idx + 1])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.bitwise_not(image)
        ret, image = cv2.threshold(image, self.thresh, 255, cv2.THRESH_BINARY)
        image = imutils.rotate_bound(image, 90)

        mask_name = os.path.join(self.root_dir, self.image_files[2 * idx])
        mask = cv2.imread(mask_name)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = imutils.rotate_bound(mask, 90)

        x, y, w, h = cv2.boundingRect(mask)
        crop_image = image[y:y + h, x:x + w]
        if crop_image.size != 0:
            image = cv2.resize(crop_image, (28, 28))
        else:
            image = cv2.resize(image, (28, 28))
        cv2.imshow('image', image)
        cv2.waitKey(10)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, 5


if __name__ == '__main__':
    # create Dataset
    dataset = MaskDataset('./')
    cv2.imshow('image', dataset.__getitem__(0)['image'])
    cv2.waitKey(5000)
