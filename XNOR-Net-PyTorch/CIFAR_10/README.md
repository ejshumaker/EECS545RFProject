# PyTorch models trained on CIFAR-10 dataset
This folder contains various networks trained on the CIFAR-10 dataset. These include both XNOR networks, where the network weights are binary and the input images are converted to binary, as well as Binary Weight Networks (BWN), which do no binarize the inputs into each layer.

## Description of Models
- `nin_best_BWN.pth.tar`:       BWN Network-in-Network Model. 87.89% Testing Accuracy.
- `vgg_best_BWN.pth.tar`:       BWN VGG13 Model. 75.81% Testing Accuracy.
- `resnet34_best_BWN.pth.tar`:  BWN ResNet34 Model. 83.92% Testing Accuracy.
- `resnet20_best_BWN.pth.tar`:  BWN ResNet20 Model. 84.86% Testing Accuracy.

- `nin_best_XNOR.pth.tar`:      XNOR Network-in-Network Model. 85.96% Testing Accuracy.
- `vgg_best_XNOR.pth.tar`:      XNOR VGG13 Model. 77.15% Testing Accuracy.
- `resnet34_best_XNOR.pth.tar`: XNOR ResNet34 Model. 75.67% Testing Accuracy.
- `resnet20_best_XNOR.pth.tar`: XNOR ResNet20 Model. 65.83% Testing Accuracy.


## How to use pretrained models
In the parent directory, follow instructions to run a pretrained model, supplying the `--arch` argument corresponding to each network. Optionally add the `--bwn` argument to avoid binarizing the inputs

Example of running evaluation on CIFAR-10 test data
```bash
cd ..
python3 main.py --cpu --arch ['nin' | 'vgg' | 'resnet20' | 'resnet34'] [--bwn] --pretrained <path_to_model> --evaluate
```

Example of running evaluation on fastMCD data
```bash
cd ..
python3 main.py --cpu --arch ['nin' | 'vgg' | 'resnet20' | 'resnet34'] [--bwn] --pretrained <path_to_model> --multi_fastMCD ../../fastMCD_output/streetlight_results --label 1
```