import torch
import torch.nn as nn


class Downsample(nn.Module):
    def __init__(self, in_planes, nd, kd=3, padding=1, stride=2):
        """
        下采样层
        Args:
            in_planes: 输入通道数
            nd: 中间层通道数
            kd: 卷积核大小
            padding: 填充
            stride: 步长
        """
        super(Downsample, self).__init__()
        # 补充材料中说明, 使用ReflectionPadding替代zero padding
        self.padder = nn.ReflectionPad2d(padding)
        # 补充材料中说明, 使用步长配合卷积实现下采样操作
        self.conv1 = nn.Conv2d(in_planes, nd, kernel_size=kd, stride=stride)
        self.bn1 = nn.BatchNorm2d(nd)

        self.conv2 = nn.Conv2d(in_channels=nd, out_channels=nd, kernel_size=kd, stride=1)
        self.bn2 = nn.BatchNorm2d(nd)

        # 补充材料中说明, 使用LeakyReLU作为非线性
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.padder(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
