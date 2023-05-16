import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class BasicConv1d(nn.Module):
    def __init__(self, i_nc, o_nc, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(i_nc, o_nc, **kwargs)
    def forward(self, x):
        x = self.conv(x)
        return F.relu_(x)

class RT_CNN(nn.Module):
    def __init__(self, flat_size, input_nc, class_num, acc_num):
        super(RT_CNN, self).__init__()
        self.accs = acc_num
        
        self.conv = BasicConv1d(input_nc, 196, kernel_size=16, stride=1, padding=8)
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)

        self.fc = nn.Sequential(
            nn.Linear(flat_size + 40*acc_num, 1024),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(1024, class_num)
        )

    def forward(self, x, stats): 
        x = self.conv(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = torch.cat([x, stats], dim=1)
        logits = self.fc(x)

        return logits



        

    
    
