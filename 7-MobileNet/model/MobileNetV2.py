import torch
import torch.nn as nn
from torchsummary import summary


class BasicConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        # 当 kernel_size = 1 时, padding = 0,
        # 当 kernel_size = 3 时, padding = 1, 此时, stride=1时图像尺寸不变; 当stride=2时, 图像尺寸减半
        padding = (kernel_size - 1) // 2
        super(BasicConv, self).__init__(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            # MobileNetV2 使用 ReLU6()
            nn.ReLU6(inplace=True)
        )


# 倒置残差结构
class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = int(in_planes * expand_ratio)
        # 论文中只体现stride==1, pytorch官方源码添加了输入通道数等于输出通道数条件
        self.use_shortcut = stride == 1 and in_planes == out_planes

        layers = []

        # 如果expand != 1 添加1x1卷积升维
        if expand_ratio != 1:
            layers.append(BasicConv(in_planes, hidden_channel, kernel_size=1))

        layers.extend([
            # 3x3 深度卷积
            BasicConv(hidden_channel, hidden_channel, stride=stride, kernel_size=3, groups=hidden_channel),
            # 1x1 逐点卷积降维
            nn.Conv2d(hidden_channel, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes)
            # 倒置残差结构最后一个1x1卷积层使用线性激活函数, 即实现时不适用激活函数
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()

        input_channel = int(32 * width_mult)
        last_channels = int(1280 * width_mult)

        inverted_residual_setting = {
            't': [1, 6, 6, 6, 6, 6, 6],
            'c': [16, 24, 32, 64, 96, 160, 320],
            'n': [1, 2, 3, 4, 3, 3, 1],
            's': [1, 2, 2, 2, 1, 2, 1]
        }

        # 第一个全卷积层
        features = []
        features.append(BasicConv(3, input_channel, kernel_size=3, stride=2))
        inverted_residual_setting['c'] = [int(width_mult * c) for c in inverted_residual_setting['c']]

        # 构建倒置残差块
        for values in zip(*inverted_residual_setting.values()):
            t, c, n, s = values
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, c, stride, expand_ratio=t))
                input_channel = c

        # 建立最后一个全卷积层
        features.append(BasicConv(input_channel, last_channels, kernel_size=1, stride=1))
        # 将上述features组合成一个层
        self.features = nn.Sequential(*features)

        # 建立分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channels, num_classes)
        )

        # 初始化权重
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


def v2(num_classes=1000, width_mult=1.):
    return MobileNetV2(num_classes=num_classes, width_mult=width_mult)
