import torch
import torch.nn as nn


class FWAMSConv(nn.Module):
    """
    FWAMSConv: Forest Wildfire Adaptive Multi-Scale Convolution
    中文：森林野火自适应多尺度卷积模块
    设计目标：提升森林野火与烟雾在极小尺度、复杂背景下的特征表达能力
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        compression_factor=2,
    ):
        super(FWAMSConv, self).__init__()

        # =========================
        # 1. 通道压缩（降低计算量）
        # =========================
        squeeze_channels = max(1, in_channels // compression_factor)
        self.squeeze = nn.Conv2d(
            in_channels,
            squeeze_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        # =========================
        # 2. 多尺度深度可分离卷积
        #    3×3 / 5×5 / 7×7
        # =========================
        self.dw_3x3 = nn.Conv2d(
            squeeze_channels,
            squeeze_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=squeeze_channels,
            bias=False
        )

        self.dw_5x5 = nn.Conv2d(
            squeeze_channels,
            squeeze_channels,
            kernel_size=5,
            stride=stride,
            padding=2,
            groups=squeeze_channels,
            bias=False
        )

        self.dw_7x7 = nn.Conv2d(
            squeeze_channels,
            squeeze_channels,
            kernel_size=7,
            stride=stride,
            padding=3,
            groups=squeeze_channels,
            bias=False
        )

        # =========================
        # 3. 共享点卷积（多尺度融合）
        # =========================
        self.pointwise = nn.Conv2d(
            squeeze_channels * 3,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        # =========================
        # 4. 归一化与激活
        # =========================
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # 通道压缩
        x = self.squeeze(x)

        # 多尺度特征提取
        x_3 = self.dw_3x3(x)   # 局部火焰细节
        x_5 = self.dw_5x5(x)   # 中尺度烟雾扩散
        x_7 = self.dw_7x7(x)   # 大尺度火灾语义

        # 多尺度拼接
        x = torch.cat([x_3, x_5, x_7], dim=1)

        # 特征融合
        x = self.pointwise(x)

        return self.act(self.bn(x))