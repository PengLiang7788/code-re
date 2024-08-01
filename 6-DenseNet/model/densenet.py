import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        out = self.conv(self.relu(self.bn1(x)))
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
    def __init__(self, block, nb_layers, in_channels, growth_rate, drop_rate=0.0):
        """
        Args:
            block: BasicBlock/BottleNeck
            nb_layers: the number of block
            in_channel: input channel
            growth_rate: growth rate
            drop_rate: dropout rate
        """
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_channels, growth_rate, nb_layers, drop_rate)
    
    def _make_layer(self, block, in_channels, growth_rate, nb_layers, drop_rate):
        layers = []
        for i in range(nb_layers):
            # 每一层的输出都是 growth_rate 个通道, 块中前面块的输出作为后面块的输入
            layers.append(block(in_channels + i * growth_rate, growth_rate, drop_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

# 适用于出ImageNet数据集之外的DenseNet结构
class DenseNet(nn.Module):
    def __init__(self, depth, num_classes=10, growth_rate=12, 
                 reduction=0.5, bottleNeck=True, drop_rate=0.0):
        """
        Args:
            depth: 模型深度
            num_classes: 数据集类别数
            growth_rate: 模型增长率
            reduction: 压缩因子
            bottleNeck: BottleNeck/BasicBlock, default BottleNeck
            drop_rate: dropout rate
        """
        super(DenseNet, self).__init__()
        # DenseNet-BC第一个卷积层输出通道数为两倍的增长率
        in_channel = 2 * growth_rate
        # 三个密集块拥有相同的层数
        # 第一个卷积层, 最后一个全连接层, 中间两个过渡层, 每个过渡层拥有一个权重层,
        # conv -> DenseBlock -> TransitionLayer -> DenseBlock -> TransitionLayer -> DenseBlock -> FullyConnected
        # 稠密块总层数为 depth - 4, 每个稠密块层数 (depth - 4) / 3
        n = (depth - 4) / 3
        if bottleNeck == True:
            # 每个BottleNeck有两个权重层
            # 稠密块中BottleNeck数量
            n = n / 2
            block = BottleNeck
        else:
            # 每个BasicBlock中只有一个权重层
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        # 输入图像 3x32x32, 每个3x3卷积使用1个像素填充0, 且保持输出特征图大小不变
        self.conv1 = nn.Conv2d(3, in_channel, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(block, n, in_channel, growth_rate, drop_rate)
        in_channel = int(in_channel + n * growth_rate)
        # DenseNet-BC 中间层生成 (θ * m)(向下取整)个输出特征图, 其中m表示上一个稠密层输出特征图数
        self.trans1 = TransitionBlock(in_channel, int(math.floor(reduction * in_channel)), drop_rate)
        in_channel = int(math.floor(reduction * in_channel))
        # 2nd block
        self.block2 = DenseBlock(block, n, in_channel, growth_rate, drop_rate)
        in_channel = int(in_channel + n * growth_rate)
        self.trans2 = TransitionBlock(in_channel, int(math.floor(reduction * in_channel)), drop_rate)
        in_channel = int(math.floor(reduction * in_channel))
        # 3rd block
        self.block3 = DenseBlock(block, n, in_channel, growth_rate, drop_rate)
        in_channel = int(in_channel + n * growth_rate)
        # 全局平均池化
        self.bn = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        # 全连接层
        self.fc = nn.Linear(in_channel, num_classes)
        self.in_channel = in_channel

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        
        out = self.trans2(self.block2(out))
        
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)
        