import torch
import torch.nn as nn


# 适用于网络层数较浅的结构，resnet-18/34
class BasicBlock(nn.Module):
    # 主分支的卷积核个数最后一层与第一层相同
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        Args:
            in_channel:  输入通道数
            out_channel: 输出通道数
            stride: 步长
            downsample: 下采样
        """
        super(BasicBlock, self).__init__()
        # stride为1进行实线残差连接, stride=2进行虚线残差连接
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                               kernel_size=3, stride=stride, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
    
    def forward(self, x):
        # 恒等映射, 如果是实线表示的残差结构, 则直接将x与最后一层卷积结果相加
        # 如果是虚线表示的残差结构, x还需使用1x1卷积匹配维度
        identity = x # 保存输入的数据, 便于后续进行残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 第一个卷积输出
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 第二个卷积
        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = self.relu(x)

        return x



# 适用于网络层数较深的结构，resnet-50/101/152
class BottleNeck(nn.Module):
    # 主分支的卷积核个数最后一层会变为第一层的四倍
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        Args:
            in_channel: 输入通道数
            out_channel: 输出通道数
            stride: 步长
            downsample: 下采样
        """
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channel=out_channel,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        # stride为1进行实线残差连接, stride=2进行虚线残差连接
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # 虚线残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 通道数压缩
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # 通道数恢复
        x = self.conv1(x)
        x = self.bn3(x)
        # 将主分支与捷径分支相加
        x += identity
        x = self.relu(x)

        return x
    


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()



