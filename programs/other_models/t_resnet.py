import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, i_nc, n_fs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(i_nc, n_fs, kernel_size=7, padding=3, stride=1, bias=False),
            nn.BatchNorm1d(n_fs),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(n_fs, n_fs, kernel_size=5, padding=2, stride=1, bias=False),
            nn.BatchNorm1d(n_fs),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(n_fs, n_fs, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(n_fs)
        )
        if i_nc == n_fs:
            self.shortcut = nn.BatchNorm1d(n_fs)
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(i_nc, n_fs, kernel_size=1, padding=0, stride=1, bias=False),
                nn.BatchNorm1d(n_fs),
            )
    def forward(self, x):
        conv_y = self.conv1(x)
        conv_y = self.conv2(conv_y)
        conv_y = self.conv3(conv_y)
        shortcut_y = self.shortcut(x)
        y = conv_y + shortcut_y
        return F.relu_(y)

class ResNet_TSC(nn.Module):
    
    def __init__(self, nc_input, n_classes):
        super(ResNet_TSC, self).__init__()
        n_fs = 64

        self.block1 = BasicBlock(nc_input, n_fs)
        self.block2 = BasicBlock(n_fs, n_fs*2)
        self.block3 = BasicBlock(n_fs*2, n_fs*2)
        self.final = nn.Linear(n_fs*2, n_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.adaptive_avg_pool1d(x,1)
        x = x.view(x.size(0), -1)
        x = self.final(x)
        return x

    
