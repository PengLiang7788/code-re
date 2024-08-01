import torch
import torch.nn as nn
import torch.nn.functional as F


# 对应论文中 Composit function
class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        # Implement Details 中指出, 所有的3x3卷积使用0填充1个像素 ==> padding=1
        # 保持特征图大小不变 ==> stride=1
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropout_rate

    # Composit function bn -> relu -> conv
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv(out)

        # Sec4.2 Training 除了第一个卷积层, 每个卷积层后加一个dropout层
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        # 将本层的输出和前面层的输入拼接起来, 作为后面层的输入
        return torch.cat([x, out], 1)            

# Poolying layers 过渡层
class TransitionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channel)
        # Table1 表示每个conv都表示一个 BN-ReLU-Conv
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                              kernel_size=1, stride=1, padding=0, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.droprate=dropout_rate

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.avgpool(out)
        # 过渡层不需要进行稠密连接
        return out
        
# BottleNeck layer 瓶颈层, 在 3x3 卷积层之前使用 1x1 卷积降维
class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0):
        super(BottleNeck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        # 使用1x1卷积降维, 生成 4k 个特征图, k=out_channel
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=4 * out_channel,
                              kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * out_channel)
        self.conv2 = nn.Conv2d(in_channels=4 * out_channel, out_channels=out_channel, 
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropout_rate
        
    def forward(self, x):
        # 所有卷积操作之前都会进行 BN 和 ReLU
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()