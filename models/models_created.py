# This file will be used to declare the different architectures for the models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision.models import resnet18, ResNet18_Weights
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



# CNN (ResNet 18) + LSTM Model
## This model uses a pre-trained ResNet-18 as the CNN backbone, followed by an LSTM layer.
# To predict the GRS, the model takes a sequence of frames as input.
class CNNLSTM(Module):
    def __init__(self, hidden_dim=256, num_classes=4, cnn_output_dim=512, bidirectional=False):
        super(CNNLSTM, self).__init__()

        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base_model.children())[:-1])  # Remove the last layer to connect to LSTM

        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)              # [B*T, 512, 1, 1]
        features = features.view(B, T, -1)  # [B, T, 512]
        lstm_out, (h_n, _) = self.lstm(features)
        out = self.classifier(h_n[-1])      # [B, num_classes]
        return out