import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from functools import partial


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


class DownSampling(nn.Module):
    def __init__(self, in_planes: int, out_planes: int):
        super(DownSampling, self).__init__()
        self.branch1 = nn.Sequential(
            # 图像长宽减半, AvgPool2d中padding默认等于kernel_size
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        self.branch2 = nn.Sequential(
            # 输入图像长宽减半, padding默认为0
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(out_planes)
        )
        self.branch3 = nn.Sequential(
            # 图像从[b, c, h, w] -> [b, c, 1, 1]
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        result = branch1 + branch2
        result = result * branch3
        return F.silu(result, inplace=True)


class FusionBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int):
        super(FusionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        # 由于在最开始有一个concatenation, 导致Fusion模块内通道数翻倍
        self.groups = in_planes
        hidden_channel = in_planes * 2
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(hidden_channel, out_planes, kernel_size=1, groups=2, stride=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(hidden_channel, out_planes, kernel_size=3, stride=2, groups=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        self.branch3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(hidden_channel, out_planes, kernel_size=1, groups=2, bias=False),
            nn.Sigmoid()
        )

    # 通道混洗
    def channel_shuffle(self, x):
        b, c, h, w = x.data.size()
        channels_per_groups = c // self.groups

        x = x.reshape(b, channels_per_groups, self.groups, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, c, h, w)

        return x

    def forward(self, input1, input2):
        # concatenation
        x = torch.cat([self.bn(input1), self.bn(input2)], dim=1)
        # 通道混洗
        x = self.channel_shuffle(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        result = branch1 + branch2
        result = result * branch3

        return F.silu(result, inplace=True)
