import torch
import torch.nn as nn


# 适用于网络层数较浅的结构，resnet-18/34
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        """
        Args:
            in_channel:  输入通道数
            out_channel: 输出通道数
            stride: 步长, stride=1对应实线表示的残差结构, 即输入特征图维度与输出维度相同
                    stride=2对应虚线表示的残差结构, 需要使用1x1卷积匹配维度
            downsample: 下采样
        """
        super(BasicBlock, self).__init__()
        # 卷积操作保证输入尺寸与输出尺寸保持不变
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
        identity = x
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
    expansion = 4
    def __init__(self):
        super(BottleNeck, self).__init__()



class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()



