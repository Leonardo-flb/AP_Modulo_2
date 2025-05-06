
# This file will be used to declare the different architectures for the models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.nn import Sequential
from torch.nn import Linear
from torch.nn import Flatten
from torch.nn import BatchNorm1d
from torch.nn import Conv3d
from torch.nn import MaxPool3d
from torch.nn import BatchNorm3d
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Dropout
from torch.nn import Softmax, Tanh
from torch.nn import Module
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


class ResNet3D(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet3D, self).__init__()
        
        resnet2d = models.resnet18(pretrained=True)
        
        self.inplanes = 64
        self.conv1 = Conv3d(3, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        
        self.bn1 = resnet2d.bn1
        self.relu = resnet2d.relu
        self.maxpool = resnet2d.maxpool
        self.layer1 = self._make_layer(resnet2d.layer1[0], 64)
        self.layer2 = self._make_layer(resnet2d.layer2[0], 128)
        self.layer3 = self._make_layer(resnet2d.layer3[0], 256)
        self.layer4 = self._make_layer(resnet2d.layer4[0], 512)
        
        self.fc = Linear(512, num_classes)

    def _make_layer(self, block, planes):
        layers = []
        layers.append(block(self.inplanes, planes, stride=2))
        self.inplanes = planes
        return Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = Flatten(x, 1)
        x = self.fc(x)
        return x
