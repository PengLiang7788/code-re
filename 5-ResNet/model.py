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
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        # 恒等映射, 如果是实线表示的残差结构, 则直接将x与最后一层卷积结果相加
        # 如果是虚线表示的残差结构, x还需使用1x1卷积匹配维度
        identity = x  # 保存输入的数据, 便于后续进行残差连接
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

        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channel)

        # stride为1进行实线残差连接, stride=2进行虚线残差连接
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False,
        )
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
        x = self.conv3(x)
        x = self.bn3(x)
        # 将主分支与捷径分支相加
        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, block_num, num_classes=1000, include_top=True):
        """
        Args:
            block: 对应网络选取 BasicBlock/BottleNeck
            block_num: (list)各个block的数量的列表, 如resnet-34中为[3, 4, 6, 3]
            num_classes: 分类数
            include_top: 分类头, 线性层
        """
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.in_channel,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 创建四个残差层
        # 第一个block输入维度都相同, 即stride均为1
        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化卷积层权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.bn1(x)
        # N x 64 x 112 x 112
        x = self.relu(x)
        # N x 64 x 112 x 112
        x = self.maxpool(x)
        # N x 64 x 56 x 56
        x = self.layer1(x)
        # N x (64 * expansion) x 28 x 28
        x = self.layer2(x)
        # N x (128 * expansion) x 14 x 14
        x = self.layer3(x)
        # N x (256 * expansion) x 7 x 7
        x = self.layer4(x)
        # N x (512 * expansion) x 7 x 7

        if self.include_top:
            x = self.avgpool(x)
            # N x (512 * expansion) x 1 x 1
            x = torch.flatten(x, 1)
            # N x (512 * expansion)
            x = self.fc(x)
            # N x numclasses

        return x

    # 创建一个残差层
    def _make_layer(self, block, channel, block_num, stride=1):
        """
        Args:
            block: 残差块, BasicBlock/BottleNeck
            channel: 残差结构中第一个卷积层卷积核数量
            block_num: 该层包含多少个残差结构
            stride: 步长, 除第一个残差结构外, 所有残差结构的第一个卷积层的步长为2
        """
        downsample = None
        # 步长不为1, 说明是该层的第一个残差结构, 需要进行下采样
        # 输入通道数不等于该层第一个卷积层通道数*扩张因子说明是该层的第一个残差结构
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channel,
                    channel * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channel * block.expansion),
            )

        # 存储残差结构
        layers = []
        layers.append(
            block(self.in_channel, channel, stride=stride, downsample=downsample)
        )

        # 特征图已经经过了一次残差结构
        self.in_channel = channel * block.expansion

        # 实线残差结构
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)


def resnet18(num_classes=1000, include_top=True, pretrained=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, include_top)


def resnet34(num_classes=1000, include_top=True, pretrained=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, include_top)


def resnet50(num_classes=1000, include_top=True, pretrained=False):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes, include_top)


def resnet101(num_classes=1000, include_top=True, pretrained=False):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes, include_top)


def resnet152(num_classes=1000, include_top=True, pretrained=False):
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes, include_top)


if __name__ == "__main__":
    model = resnet50()
    x = torch.randn((32, 3, 224, 224))
    y = model(x)
    print(y)
