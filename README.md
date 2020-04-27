# EECS545 Project: Classification of Moving Objects via Change Detection and XNOR Networks in Real Time

## Usage:
Compile FastestMCD using Visual Studio, and run it on a dataset to produce images and masks OR use the supplied ones we already ran it on.
```bash
cd <REPO ROOT>/fastMCD_output
$ ./fastMCD ../Data/streetlight.mp4 1
```

Run fastMCD output through XNOR Network-In-Network Classifier. This outputs a file of bounding box locations and object classifications for each frame.
```bash
$ cd <REPO ROOT>/XNOR-Net-PyTorch/CIFAR10
$ python3 main.py --cpu --pretrained models/nin_best_XNOR.pth.tar --multi_fastMCD ../../fastMCD_output/streetlight_results --label 1
```

Run fastMCD output through Resnet34 (requires download). This outputs a file of bounding box locations and object classifications for each frame.
```bash
$ cd <REPO ROOT>/XNOR-Net-PyTorch/PyTorch_CIFAR10
$ python3 cifar10_download.py
$ python3 cifar10_test.py --classifier resnet34 --fastMCD ../../fastMCD_output/streetlight_results --label 1
```

TODO: Eric add your evaluation results here
```bash
```



### MNIST Proof Of Concept
To use fastMCD with the xnor net, use the following basic procedure (use for MNIST case):

1) Build fastMCD:
```bash
    cd fastMCD
    mkdir build
    cd build
    cmake ..
    make
    mv fastMCD ../../fastMCD_output
```

2) Run fastMCD on number.m4v
```bash
    cd fastMCD_output
    ./fastMCD ../Data/number.m4v 1
```

3) Run xnor net on images
```bash
    cd XNOR-Net-PyTorch/MNIST
    python3 main.py --pretrained models/LeNet_5.best.pth.tar --fastMCD ../fastMCD_output/results/
```

