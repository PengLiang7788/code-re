import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchsummary import summary


class Downsample(nn.Module):
    def __init__(self, in_planes, nd, kd=3, padding=1, stride=2):
        """
        下采样层
        Args:
            in_planes: 输入通道数
            nd: 中间层通道数
            kd: 卷积核大小
            padding: 填充, 默认为1, 这样可以使得当stride=2时, 图像尺寸减半
            stride: 步长, 默认为2, 下采样操作, 图像尺寸减半
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


class SkipConnection(nn.Module):
    def __init__(self, in_channels, ns=4, stride=1, ks=1, padding=0):
        """
        Args:
            in_channels: 输入通道数
            stride: 步长
            ns: 中间层通道数
            ks: 卷积核大小
        """
        super(SkipConnection, self).__init__()
        self.conv = nn.Conv2d(in_channels, ns, kernel_size=ks, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(ns)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class DIPConfig:
    def __init__(self, depth: int = 5, in_channels=3, nu: List = [8, 16, 32, 64, 128],
                 nd: List = None, ku: List = [3, 3, 3, 3, 3], kd: List = None,
                 ns: List = [0, 0, 0, 4, 4], ks: List = [None, None, None, 1, 1],
                 mode='bilinear'):
        self.depth = depth

        assert depth == len(nu), 'Hyperparameters do not match network depth.'
        self.in_channels = in_channels
        self.nu = nu
        self.nd = nd if nd is not None else nu
        self.ku = ku
        self.kd = kd if kd is not None else ku
        self.ns = ns
        self.ks = ks
        self.mode = mode


class DIP(nn.Module):
    def __init__(self, config: DIPConfig):
        super(DIP, self).__init__()
        self.depth = config.depth
        self.config = config

        # 下采样层, 输出通道记录在nd中, 卷积核大小记录在kd中
        self.downs = nn.ModuleList(
            [Downsample(in_planes=config.nd[i - 1], nd=config.nd[i], kd=config.kd[i]) if i != 0 else
             Downsample(in_planes=config.in_channels, nd=config.nd[i], kd=config.kd[i]) for i in range(config.depth)])

        self.skips = nn.ModuleList()
        for i in range(config.depth):
            if config.ks[i] is not None:
                # 只有当kernel_size不为空时, 才会有条约连接
                self.skips.append(SkipConnection(in_channels=config.nd[i], ns=config.ns[i], ks=config.ks[i]))

        self.ups = nn.ModuleList()
        for i in range(config.depth - 1, -1, -1):
            if i != config.depth - 1:
                # 除最后一层外, 其他层输入通道数为前一层的输出通道数+对应下采样层进行跳跃连接的输出层通道数
                self.ups.append(
                    UpSample(in_channels=config.nu[i + 1] + config.ns[i],nu=config.nu[i] ,ku=config.ku[i], mode=config.mode))
            else:
                self.ups.append(UpSample(in_channels=config.ns[i], nu=config.nu[i], ku=config.ku[i], mode=config.mode))

        self.conv_out = nn.Conv2d(config.nu[0], config.in_channels, 1, padding=0)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        # 跳跃连接
        s = []
        # 跳跃连接数
        ns = 0

        # 下采样
        for i in range(self.depth):
            x = self.downs[i].forward(x)
            if self.config.ks[i] is not None:
                s.append(self.skips[ns].forward(x))
                ns += 1

        # 上采样
        for i in range(self.depth):
            if i == 0 and self.config.ks[self.depth - i - 1] is not None:
                x = self.ups[i].forward(s[-1])
            elif i != 0 and self.config.ks[self.depth - i - 1] is not None:
                x = self.ups[i].forward(torch.cat([x, s[self.depth - i - 1]], dim=1))
            elif i != 0 and self.config.ks[self.depth - i - 1] is None:
                x = self.ups[i].forward(x)
            else:
                pass

        x = self.sigm(self.conv_out(x))

        return x


def super_resolution(in_channels: int = 3):
    nu = [128, 128, 128, 128, 128]
    nd = nu
    kd = [3, 3, 3, 3, 3]
    ku = kd
    ns = [4, 4, 4, 4, 4]
    ks = [1, 1, 1, 1, 1]
    mode = 'bilinear'
    config = DIPConfig(in_channels=in_channels, nu=nu, nd=nd, ku=ku, kd=kd, ns=ns, ks=ks, mode=mode)
    net = DIP(config)
    return net


def inpainting(in_channels: int = 3):
    nu = [16, 32, 64, 128, 128, 128]
    nd = nu
    kd = [3, 3, 3, 3, 3, 3]
    ku = [5, 5, 5, 5, 5, 5]
    ns = [0, 0, 0, 0, 0, 0]
    ks = [None, None, None, None, None, None]
    mode = 'nearest'
    config = DIPConfig(in_channels=in_channels, nu=nu, nd=nd, ku=ku, kd=kd, ns=ns, ks=ks, mode=mode)
    net = DIP(config)
    return net


def denoising(in_channels: int = 1):
    nu = [8, 16, 32, 64, 128]
    nd = nu
    kd = [3, 3, 3, 3, 3]
    ku = kd
    ns = [0, 0, 0, 4, 4]
    ks = [None, None, None, 1, 1]
    mode = 'bilinear'
    config = DIPConfig(in_channels=in_channels, nu=nu, nd=nd, ku=ku, kd=kd, ns=ns, ks=ks, mode=mode)
    net = DIP(config)
    return net


if __name__ == '__main__':
    net = denoising()
    x = torch.randn((1, 1, 180, 180))
    summary(net, x)