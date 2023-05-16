import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

class FCN_TSC(nn.Module):
    def __init__(self, i_nc, n_classes, segment_size):
        super(FCN_TSC, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(i_nc, 128, kernel_size=7, padding=3, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2, stride=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        
        self.final = nn.Linear(128, n_classes)

    def forward(self, x): 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.adaptive_avg_pool1d(x,1)
        x = x.view(x.size(0), -1)
        logits = self.final(x)

        return logits

    
