import torch
from torch import nn


# Hybrid pooling attention (HPAttention)
class HPAttention(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(HPAttention, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.map = nn.AdaptiveMaxPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Y avg
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # X avg
        self.max_h = nn.AdaptiveMaxPool2d((None, 1))  # Y avg
        self.max_w = nn.AdaptiveMaxPool2d((1, None))  # X avg

        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w --->2048,2,11,11
        x_h = self.pool_h(group_x)  # 2048,2,11,1
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # 2048,2,1,11--->2048,2,11,1
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # 2048,2,22,1
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # 2048,2,11,1
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # 2048,2,11,11
        x2 = self.conv3x3(group_x)  # 2048,2,11,11

        y_h = self.max_h(group_x)  # 2048,2,11,1
        y_w = self.max_w(group_x).permute(0, 1, 3, 2)
        yhw = self.conv1x1(torch.cat([y_h, y_w], dim=2))  # 2048,2,22,1
        y_h, y_w = torch.split(yhw, [h, w], dim=2)  # 2048,2,11,1
        y1 = self.gn(group_x * y_h.sigmoid() * y_w.permute(0, 1, 3, 2).sigmoid())  # 2048,2,11,11
        y11 = y1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw 2048,2,121
        y12 = self.softmax(self.map(y1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 2048,1,2

        x11 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw 2048,2,121
        x12 = self.softmax(
            self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 2048,2,1,1-->2048,2,1--->2048,1,2
        x21 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw #2048,2,121
        x22 = self.softmax(
            self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 2048,2,1,1-->2048,2,1--->2048,1,2
        weights = (torch.matmul(x12, y11) + torch.matmul(y12, x11)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)



