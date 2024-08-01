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
                              kernel_size=3, stride=1, padding=1)
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

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()