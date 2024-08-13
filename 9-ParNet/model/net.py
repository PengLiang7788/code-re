import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from functools import partial


class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            # 不深的网络没有足够的非线性, 限制了其表征能力, 本文使用SiLU替代了ReLU激活
            activation_layer = nn.SiLU
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )


class SSEBlock(nn.Module):
    def __init__(self, in_planes: int):
        # SSE模块一个分支只进行BatchNorm, 输入特征矩阵形状不会改变
        # SSE另一个分支会先进行全局平均池化, 然后使用1x1卷积, 其中输出通道数仅受1x1卷积影响
        # SSE模块最好需要将两个分支的输出结果进行相乘再输出, 为保证输出结果相同, 1x1卷积的输入通道数应当等于输出通道数
        super(SSEBlock, self).__init__()
        self.norm = nn.BatchNorm2d(in_planes)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1, bias=False)

    def forward(self, x):
        bn = self.norm(x)
        output = F.sigmoid(self.fc(self.pool(x)))

        return torch.mul(bn, output)


