import torch
import torch.nn as nn
import torch.nn.functional as F

def channel_shuffle(x, groups):
    batchsize, num_channels, T = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, T)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, T)

    return x

class InplaceShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_groups):
        ctx.groups_ = n_groups
        n, c, t = input.size()
        slide = c // n_groups
        left_idx = torch.tensor([i*slide for i in range(n_groups)])
        right_idx = torch.tensor([i*slide+1 for i in range(n_groups)])

        buffer = input.data.new(n, n_groups, t).zero_()
        buffer[:, :, :-1] = input.data[:, left_idx, 1:] 
        input.data[:, left_idx] = buffer
        buffer.zero_()
        buffer[:, :, 1:] = input.data[:, right_idx, :-1]
        input.data[:, right_idx] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        n_groups = ctx.groups_
        n, c, t = grad_output.size()
        slide = c // n_groups
        left_idx = torch.tensor([i*slide for i in range(n_groups)])
        right_idx = torch.tensor([i*slide+1 for i in range(n_groups)])

        buffer = grad_output.data.new(n, left_idx,t).zero_()
        buffer[:, :, 1:] = grad_output.data[:, left_idx, :-1] # reverse
        grad_output.data[:, left_idx] = buffer
        buffer.zero_()
        buffer[:, :, :-1] = grad_output.data[:, right_idx, 1:]
        grad_output.data[:, right_idx] = buffer
        return grad_output, None

class GTSConv(nn.Module):
    def __init__(self, i_nc, n_groups):
        super(GTSConv, self).__init__()
        self.groups = n_groups
        self.conv = nn.Conv1d(i_nc, i_nc, kernel_size=1, padding=0, bias=False, groups=n_groups)
        self.bn = nn.BatchNorm1d(i_nc)
    def forward(self, x):
        out = InplaceShift.apply(x, self.groups)
        out = self.conv(x)
        out = self.bn(out)
        return out
    
class GTSConvUnit(nn.Module):
    '''
    Grouped Temporal Shift (GTS) module
    '''
    def __init__(self, i_nc, n_fs, n_groups, first_grouped_conv=True):
        super(GTSConvUnit, self).__init__()

        self.groups = n_groups
        self.grouped_conv = n_groups if first_grouped_conv else 1

        self.perm = nn.Sequential(
            nn.Conv1d(i_nc, n_fs, kernel_size=1, groups=self.grouped_conv, stride=1, bias=False),
            nn.BatchNorm1d(n_fs),
        )

        self.GTSConv = GTSConv(n_fs, n_groups)
        
    def forward(self, x):
        out = F.relu(self.perm(x))
        out = self.GTSConv(out)
        out = channel_shuffle(out, self.groups)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, i_nc, n_fs, n_groups, first_grouped_conv=True, pool=False):
        super(ResidualBlock, self).__init__()

        self.i_nc = i_nc
        self.n_fs = n_fs
        self.pool = pool

        self.conv1 = GTSConvUnit(self.i_nc, self.n_fs, n_groups, first_grouped_conv)
        self.conv2 = GTSConvUnit(self.n_fs, self.n_fs, n_groups)
        self.conv3 = GTSConvUnit(self.n_fs, self.n_fs, n_groups)

        if i_nc == n_fs:
            self.shortcut = nn.BatchNorm1d(n_fs)
        else:
            self.shortcut = nn.Sequential(
                nn.Conv1d(i_nc, n_fs, kernel_size=1, padding=0, stride=1, bias=False),
                nn.BatchNorm1d(n_fs),
            )

    def forward(self, x):
        _,_,t = x.size()
        if self.pool:
            x = F.adaptive_max_pool1d(x,t//2)
        residual = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        shortcut_y = self.shortcut(residual)
        out = out + shortcut_y
        return F.relu(out)

class GTSNet(nn.Module):
    
    def __init__(self, nc_input, n_classes):
        super(GTSNet, self).__init__()
        n_fs = 128
        n_groups = 32

        first_channels = 32

        self.stem = nn.Conv1d(nc_input, first_channels, kernel_size=3, stride=1, padding=1,bias=True) #channel extansion
        self.bn = nn.BatchNorm1d(first_channels)

        self.block1 = ResidualBlock(first_channels, n_fs,n_groups, first_grouped_conv=False, pool=True)
        self.block2 = ResidualBlock(n_fs, n_fs*2,n_groups, pool=True)
        self.block3 = ResidualBlock(n_fs*2, n_fs*2,n_groups)
        self.final = nn.Linear(n_fs*2, n_classes)
        
    def forward(self, x):
        x = self.bn(self.stem(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.adaptive_avg_pool1d(x,1)
        x = x.view(x.size(0), -1)
        x = self.final(x)
        return x

    
