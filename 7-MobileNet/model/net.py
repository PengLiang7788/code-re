import torch
import torch.nn as nn

# 深度可分离卷积层
class DepthWiseSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        深度可分离卷积模块, 包括深度卷积和逐点卷积两个部分
        深度卷积: 对输入特征矩阵每个通道使用单个卷积核提取特征
        逐点卷积: 使用1x1卷积核对深度卷积后产生的特征矩阵进行特征融合生成新的特征矩阵
        """
        super(DepthWiseSeparable, self).__init__()
        # 深度卷积部分
        # 对于输入矩阵的每个通道使用一个卷积核, 因此输出通道数等于输入通道数
        # 设置padding=1可以使得在stride=1时保持尺寸不变, 在stride=2时使得尺寸减半
        # groups=in_channel, 将输入特征图分成in_channel组，每个卷积核对应一组, 因此可以实现每个卷积核作用在一个通道上
        # 此时参数量为 输出通道数 x 卷积核大小(3 x 3)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                               padding=1, groups=in_channels, bias=False)
        # 在深度卷积和逐点卷积之后都跟着BatchNorm和ReLU
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.relu(self.bn1(x))

        x = self.pointwise(x)
        x = self.relu(self.bn2(x))

        return x

