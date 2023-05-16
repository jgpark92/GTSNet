from sys import modules
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

class BasicConv1d(nn.Module):
    def __init__(self, i_nc, o_nc, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(i_nc, o_nc, **kwargs)
    def forward(self, x):
        x = self.conv(x)
        return F.relu_(x)

class RT_CNN(nn.Module):
    def __init__(self, input_nc, class_num, segment_size):
        super(RT_CNN, self).__init__()

        n_fs = 32

        self.conv1 = BasicConv1d(input_nc, n_fs, kernel_size=7, stride=1, padding=3)

        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)

        length = math.floor((segment_size + 2) /3)

        self.conv2 = BasicConv1d(n_fs, n_fs, kernel_size=7, stride=1, padding=3)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)

        length = math.floor((length + 2) / 3) 

        self.conv3 = BasicConv1d(n_fs, n_fs, kernel_size=7, stride=1, padding=3)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)

        length = math.floor((length + 2) / 3) 

        self.fc = nn.Sequential(
            nn.Linear(n_fs*length, 512),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(512, class_num)
        )

    def forward(self, x): 
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)

        return logits

    
