import torch.nn as nn
import torch


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class LCA(nn.Module):
    def __init__(self, input_channel, reduction=32):
        super(LCA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(input_channel, input_channel, kernel_size=1,
                               stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(input_channel, input_channel, kernel_size=1,
                                stride=1, padding=0, groups=input_channel)
        self.conv_w = nn.Conv2d(input_channel, input_channel, kernel_size=1,
                                stride=1, padding=0, groups=input_channel)

    def forward(self, x):  # torch.Size([2, 32, 64, 64])
        identity = x

        b, c, h, w = x.size()
        x_h = self.pool_h(x)  # torch.Size([2, 32, 64, 1])
        x_w = self.pool_w(x)  # torch.Size([2, 32, 1, 64])

        a_h = self.conv_h(x_h).sigmoid()  # torch.Size([2, 32, 64, 1])
        a_w = self.conv_w(x_w).sigmoid()  # torch.Size([2, 32, 1, 64])

        out = identity * a_w * a_h  # torch.Size([2, 32, 64, 64])

        return out


