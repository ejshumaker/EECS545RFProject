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
    def __init__(self, input_channels, output_channels,
                 kernel_size=-1, stride=-1, padding=-1,
                 groups=1, dropout=0, bias=True, Linear=False, relu=True, bwn=False):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.doRelu = relu
        self.bwn = bwn

        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.conv = nn.Conv2d(input_channels, output_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, groups=groups, bias=bias)
            self.bn = nn.BatchNorm2d(output_channels, eps=1e-4, momentum=0.1, affine=True)
        else:
            self.linear = nn.Linear(input_channels, output_channels)
            self.bn = nn.BatchNorm1d(output_channels, eps=1e-4, momentum=0.1, affine=True)
        if self.doRelu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if not self.bwn:
            x, mean = BinActive()(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
        x = self.bn(x)
        if self.doRelu:
            x = self.relu(x)
        return x


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', bwn=False):
        super(BasicBlock, self).__init__()
        self.conv_bn_relu1 = BinConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, bwn=bwn)
        self.conv_bn2 = BinConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, relu=False, bwn=bwn)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = BinConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, padding=0, bias=False, relu=False, bwn=bwn)

    def forward(self, x):
        out = self.conv_bn_relu1(x)
        out = self.conv_bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bwn=False):
        super(ResNet20, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, bwn=bwn)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, bwn=bwn)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, bwn=bwn)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, bwn=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bwn=bwn))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             m.weight.data.normal_(0, 0.05)
    #             m.bias.data.zero_()