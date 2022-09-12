import torch
import torchvision
import numpy
from torch.nn import Conv2d, Module, Linear, MaxPool2d, ReLU, LogSoftmax
from torch import flatten

class ConvNet(Module):
    def __init__(self, num_channels, classes):
        super(ConvNet, self).__init__()
        
        self.conv1 = Conv2d(in_channels=num_channels, out_channels=20, 
            kernel_size=(5,5))
        
        self.relu1 = ReLU()
        self.Maxpool1 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv2 = Conv2d(in_channels=20, out_channels=50, 
            kernel_size=(5,5))
        
        self.relu2 = ReLU()
        self.Maxpool2 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv3 = Conv2d(in_channels=50, out_channels=80, 
            kernel_size=(5,5))
        
        self.relu3 = ReLU()
        self.Maxpool3 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv4 = Conv2d(in_channels=80, out_channels=150, 
            kernel_size=(5,5))
        
        self.relu4 = ReLU()
        self.Maxpool4 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

        self.fc1 = Linear(in_features=86400, out_features=80)
        self.relu3 = ReLU()

        self.fc2 = Linear(in_features=80, out_features=classes)
        self.LogSoftmax = LogSoftmax(dim=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.Maxpool1(x)

        x = self.conv2(x)
        x=  self.relu2(x)
        x = self.Maxpool2(x)

        x = self.conv3(x)
        x=  self.relu3(x)
        x = self.Maxpool3(x)

        x = self.conv4(x)
        x=  self.relu4(x)
        x = self.Maxpool4(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        output = self.LogSoftmax(x)

        return output