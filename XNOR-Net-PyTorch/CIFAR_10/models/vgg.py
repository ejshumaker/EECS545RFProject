import torch.nn as nn
import torch
import torch.nn.functional as F


class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        input = input.sign()
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=-1, stride=-1,
                 padding=-1, groups=1, dropout=0, Linear=False, bwn=False):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.bwn = bwn

        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(input_channels, output_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, groups=groups)
        else:
            self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        if not self.bwn:
            x, mean = BinActive()(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
        x = self.relu(x)
        return x


# [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
class VGG13(nn.Module):
    def __init__(self, num_classes=10, init_weights=True, bwn=False):
        super(VGG13, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),
            BinConv2d(64, 64, kernel_size=3, stride=1, padding=1, bwn=bwn),  # Does Batch Norm and ReLU
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),

            BinConv2d(64, 128, kernel_size=3, stride=1, padding=1, bwn=bwn),
            BinConv2d(128, 128, kernel_size=3, stride=1, padding=1, bwn=bwn),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(128, 256, kernel_size=3, stride=1, padding=1, bwn=bwn),
            BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, bwn=bwn),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(256, 512, kernel_size=3, stride=1, padding=1, bwn=bwn),
            BinConv2d(512, 512, kernel_size=3, stride=1, padding=1, bwn=bwn),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BinConv2d(512, 512, kernel_size=3, stride=1, padding=1, bwn=bwn),
            BinConv2d(512, 512, kernel_size=3, stride=1, padding=1, bwn=bwn),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            BinConv2d(512 * 7 * 7, 4096, Linear=True, bwn=bwn),
            BinConv2d(4096, 4096, dropout=0.5, Linear=True, bwn=bwn),
            # nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()

    def forward(self, x):
        # for m in self.modules():
        #     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        #         if hasattr(m.weight, 'data'):
        #             m.weight.data.clamp_(min=0.01)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.classifier(x)
        return x
        x = x.view(x.size(0), 10)
        return x
