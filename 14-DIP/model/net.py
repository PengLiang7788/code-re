import torch
import torch.nn as nn
import torch.nn.functional as F



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


class UpSample(nn.Module):
    def __init__(self, in_channels, nu, ku=3, mode='bilinear'):
        """
        Args:
            in_channels: 输入通道
            nu: 中间层通道数
            ku: 卷积核大小
            mode: 上采样层合并模式, 去噪任务默认 bilinear
        """
        super(UpSample, self).__init__()
        # 上采样过程先BN
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.padder = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_channels, nu, kernel_size=ku, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(nu)

        self.conv2 = nn.Conv2d(nu, nu, kernel_size=ku, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(nu)

        self.relu = nn.LeakyReLU()
        self.mode = mode

    
    def forward(self, x):
        x = self.bn1(x)

        x = self.padder(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = F.interpolate(x, scale_factor=2, mode=self.mode)

        return x
