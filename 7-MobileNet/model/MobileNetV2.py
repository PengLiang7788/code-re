import torch
import torch.nn as nn
from torchsummary import summary


# 基础卷积操作, 后面跟着BatchNorm和ReLU6
class BasicConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(BasicConv, self).__init__(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


# 深度可分离卷积
class DWSConv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(DWSConv, self).__init__()
        # 深度卷积部分
        self.depthwise = BasicConv(in_planes, in_planes, kernel_size=3, stride=stride, groups=in_planes)

        # 逐点卷积部分
        self.pointwise = BasicConv(in_planes, out_planes, kernel_size=1, stride=1, groups=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand=6):
        super(BottleNeck, self).__init__()
        # 是否使用快捷连接
        # 原论文中指出当stride=1时使用shorcut, 但是pytorch官方实现版本加了输入通道数等于输出通道数时使用shortcut
        self.shortcut = stride == 1 and in_channels == out_channels

        hidden_channels = in_channels * expand

        self.conv1 = BasicConv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.dwsconv = DWSConv(hidden_channels, hidden_channels, stride=stride)
        # 最后一个卷积层, 使用线性激活函数, 即不使用激活函数
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.dwsconv(out)
        out = self.conv2(out)
        if self.shortcut:
            out += identity

        return out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, ):
        super(MobileNetV2, self).__init__()
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult)

        bottleneck_setting = {
            't': [1, 6, 6, 6, 6, 6, 6],
            'c': [16, 24, 32, 64, 96, 160, 320],
            'n': [1, 2, 3, 4, 3, 3, 1],
            's': [1, 2, 2, 2, 1, 2, 1]
        }

        features = []

        bottleneck_setting['c'] = [int(c * width_mult) for c in bottleneck_setting['c']]

        features.append(BasicConv(3, input_channel, kernel_size=3, stride=2))

        # 建立BottleNeck
        for values in zip(*bottleneck_setting.values()):
            t, c, n, s = values
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(BottleNeck(input_channel, out_channels=c, stride=stride, expand=t))
                input_channel = c

        # 建立最后一个卷积层
        features.append(BasicConv(input_channel, last_channel, 1))
        # 合并特征层
        self.features = nn.Sequential(*features)

        # 构建分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def v2(num_classes=1000, width_mult=1.0):
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult)
