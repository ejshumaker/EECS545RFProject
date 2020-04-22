import cv2
# import imutils
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from timer import Timer
from multiBlobDetector import multiObjectFrame


class FastMCD_MultiObject(Dataset):
    '''
    FastMCD Images to dataset for alexnet
    '''
    def __init__(self, root_dir, lazylabel=3, im_size=256):
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
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[1. / 255., 1. / 255., 1. / 255.])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transforms = transforms.Compose([
            transforms.Resize((im_size, im_size)),
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
        image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

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
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0))
            cv2.imshow('image w/ roi', image)
            cv2.imshow('mask', mask)
            cv2.imshow('roi', roi)
            cv2.waitKey(20)

            roi = self.transforms(Image.fromarray(roi))
            print(roi.shape)
            
            # label everything as a cat (3)
            label = 0
            if self.lazylabel:
                label = self.lazylabel
            data_label_list.append((roi, label))

        self.preprocess_timer.end_time()
        return data_label_list
