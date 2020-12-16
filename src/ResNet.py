import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm

from torch.autograd import Variable
import sys

def Conv2d(in_channels, out_channels, stride=1, kernel_size=3, padding=1):
    return weightNorm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True))

def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2,2,2,2]),
        '34': (BasicBlock, [3,4,6,3]),
        '50': (Bottleneck, [3,4,6,3]),
        '101':(Bottleneck, [3,4,23,3]),
        '152':(Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]

class normed_conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super(normed_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class TReLU(nn.Module):
    def __init__(self):
        super(TReLU, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0)
        
    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, type, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        if type == "actor":
            conv = normed_conv
            relu = nn.ReLU()
        else:
            conv = Conv2d
            relu = TReLU()
        self.conv1 = conv(in_channels, out_channels, stride) 
        self.conv2 = conv(out_channels, out_channels)
        self.relu1 = relu
        self.relu2 = relu

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                conv(in_channels, self.expansion*out_channels, stride=stride, kernel_size=1, padding=0)
            )

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, type, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        if type == "actor":
            conv = normed_conv
            relu = nn.ReLU()
        else:
            conv = Conv2d
            relu = TReLU()
        self.conv1 = conv(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2 = conv(out_channels, out_channels, kernel_size=3, stride=stride)
        self.conv3 = conv(out_channels, self.expansion*out_channels, kernel_size=1, padding=0)
        self.relu1 = relu
        self.relu2 = relu
        self.relu3 = relu

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            if type == "actor":
                self.shortcut = nn.Sequential(
                    (nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, bias=False)),
                )
            else:
                self.shortcut = nn.Sequential(
                    Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, padding=0),
                )

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = self.relu3(out)

        return out
    

class ResNet(nn.Module):
    def __init__(self, type, num_inputs, depth, num_outputs):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.type = type

        block, num_blocks = cfg(depth)

        if self.type == "actor":
            conv = normed_conv
            relu = nn.ReLU()
        else:
            conv = Conv2d
            relu = TReLU()

        self.conv1 = conv(num_inputs, 64, 2)
        self.relu1 = relu
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512, num_outputs)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.type, self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.type == "actor":
            x = torch.sigmoid(x)
        return x