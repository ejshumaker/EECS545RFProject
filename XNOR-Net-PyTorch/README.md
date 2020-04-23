# XNOR-Net-Pytorch
This is an adaptation of a [PyTorch implementation](https://github.com/jiecaoyu/XNOR-Net-PyTorch) of the [XNOR-Net](https://github.com/allenai/XNOR-Net) for an end-to-end moving object detector and classifier, using fastMCD for region proposals, and one of the below for classification.

# Usage:

## MNIST
To run this on the output of the fastMCD algorithm, train the model (see below), and make sure images are in `<path-to-results>`:
```bash
$ cd <Repository Root>/XNOR-Net-PyTorch/MNIST/
$ python3 main.py --pretrained models/LeNet_5.best.pth.tar --fastMCD <path-to-results>
```

To train your own model
Torchvision MNIST Dataset: [torchvision](https://github.com/pytorch/vision). To run the training:
```bash
$ cd <Repository Root>/XNOR-Net-PyTorch/MNIST/
$ python3 main.py
```
OR user the [pretrained model](https://drive.google.com/open?id=0B-7I62GOSnZ8R3Jzd0ozdzlJUk0). To evaluate the pretrained model:
```bash
$ cp <Pretrained Model> <Repository Root>/XNOR-Net-PyTorch/MNIST/models/
$ python3 main.py --pretrained models/LeNet_5.best.pth.tar --evaluate
```


## CIFAR-10
To run this on the output of the fastMCD algorithm, download the [pretrained model](https://drive.google.com/open?id=0B-7I62GOSnZ8UjJqNnR1V0dMbWs), and make sure images are in `<path-to-results>`:
```bash
$ cd <Repository Root>/XNOR-Net-PyTorch//CIFAR_10/
$ python3 main.py --cpu --pretrained models/nin.best.pth.tar --fastMCD <path-to-results>
```

UPDATE: To pipe the fastMCD proposed regions with multiple objects run this on the pretrained model (coming soon) on VGG network or Network in Network. The same label is used for every image frame. Note, specifying label 1 includes label 9 as correct, since the two classes are 'car' and 'truck'.
```bash
$ cd <Repository Root>/XNOR-Net-PyTorch/CIFAR_10/
$ python3 main.py --cpu --arch ['vgg' | 'nin'] --pretrained <path to model binary> --multi_fastMCD <path-to-results> --label [0 - 9]
```

UPDATE: To pipe proposed the fastMCD proposed regions with multiple objects into non-binarized networks (Note, specifying label 1 includes label 9 as correct, since the two classes are 'car' and 'truck'), run:
```bash
$ cd <Repository Root>/XNOR-Net-PyTorch/PyTorch_CIFAR10/
$ python cifar10_download.py
$ python3 cifar10_test.py --classifier ['vgg11_bn' | 'resnet34' | ...] --fastMCD --data <path-to-results> --label [0 - 9]
```

<!--
## ImageNet
DAK TODO. Below is just stuff from the original repo
### Dataset

The training supports [torchvision](https://github.com/pytorch/vision).

If you have installed [Caffe](https://github.com/BVLC/caffe), you can download the preprocessed dataset [here](https://drive.google.com/uc?export=download&id=0B-7I62GOSnZ8aENhOEtESVFHa2M) and uncompress it. 
To set up the dataset:
```bash
$ cd <Repository Root>/ImageNet/networks/
$ ln -s <Datasets Root> data
```

### AlexNet
To train the network:
```bash
$ cd <Repository Root>/ImageNet/networks/
$ python main.py # add "--caffe-data" if you are training with the Caffe dataset
```
The pretrained models can be downloaded here: [pretrained with Caffe dataset](https://drive.google.com/open?id=0B-7I62GOSnZ8bUtZUXdZLVBtUDQ); [pretrained with torchvision](https://drive.google.com/open?id=1NiVSo3K4c_kcRP10bUCirjHX5_pvylNb). To evaluate the pretrained model:
```bash
$ cp <Pretrained Model> <Repository Root>/ImageNet/networks/
$ python main.py --resume alexnet.baseline.pth.tar --evaluate # add "--caffe-data" if you are training with the Caffe dataset
```
The training log can be found here: [log - Caffe dataset](https://raw.githubusercontent.com/jiecaoyu/XNOR-Net-PyTorch/master/ImageNet/networks/log.baseline); [log - torchvision](https://github.com/jiecaoyu/XNOR-Net-PyTorch/blob/master/ImageNet/networks/log.pytorch.wd_3e-6).

## Todo
- NIN for ImageNet.

## Notes
### Gradients of scaled sign function
In the paper, the gradient in backward after the scaled sign function is  
  
![equation](http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20W_i%7D%3D%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20%7B%5Cwidetilde%7BW%7D%7D_i%7D%20%28%5Cfrac%7B1%7D%7Bn%7D+%5Cfrac%7B%5Cpartial%20sign%28W_i%29%7D%7B%5Cpartial%20W_i%7D%5Ccdot%20%5Calpha%20%29)

< !--
\frac{\partial C}{\partial W_i}=\frac{\partial C}{\partial {\widetilde{W}}_i} (\frac{1}{n}+\frac{\partial sign(W_i)}{\partial W_i}\cdot \alpha )
-- >

However, this equation is actually inaccurate. The correct backward gradient should be

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20W_%7Bi%7D%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Ccdot%20sign%28W_%7Bi%7D%29%20%5Ccdot%20%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%5B%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20%5Cwidetilde%7BW%7D_j%7D%20%5Ccdot%20sign%28W_j%29%5D%20&plus;%20%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20%5Cwidetilde%7BW%7D_i%7D%20%5Ccdot%20%5Cfrac%7Bsign%28W_i%29%7D%7BW_i%7D%20%5Ccdot%20%5Calpha)

Details about this correction can be found in the [notes](notes/notes.pdf) (section 1).
-->
