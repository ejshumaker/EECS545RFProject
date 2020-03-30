# EECS545 Project: fast object classification using fastMCD for region proposal
To use fastMCD with the xnor net, use the following basic procedure (use for MNIST case):

1) Build fastMCD:
```bash
    cd fastMCD
    mkdir build
    cd build
    cmake ..
    make
    mv fastMCD ../test
```

2) Run fastMCD on number.m4v
```bash
    cd fastMCD/test
    ./fastMCD number.m4v 1
```

3) Run xnor net on images
```bash
    cd XNOR-Net-PyTorch/MNIST
    python3 main.py --pretrained models/LeNet_5.best.pth.tar --fastMCD ../fastMCD/test/results/
```

