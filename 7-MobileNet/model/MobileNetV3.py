import torch
import torch.nn as nn
import torchvision.models
from torchsummary import summary
from torch.nn import functional as F
from typing import List
from functools import partial


class BasicBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, groups=1, norm_layer=None,
                 activation_layer=None):
        # 论文中最后一两个卷积层没有使用BN, 但是在官方实现代码中, 均使用了BN层
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6

        # 满足当stride=1时, 图片尺寸不变; 当stride=2时, 图片尺寸减半
        padding = (kernel_size - 1) // 2
        super(BasicBlock, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, in_planes, reduction=4):
        """
        SE 通道注意力机制
        Args:
            in_planes: 输入通道数
            reduction: 第一个全连接层通道数下降的倍数
        """
        super(SqueezeExcitation, self).__init__()
        # 第一个全连接层, 将通道数下降为原来的四分之一
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, kernel_size=1, stride=1, padding=0, bias=False)
        # 第二个全连接层, 恢复通道数
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # squeeze [b, c, h, w] ==> [b, c, 1, 1]
        output = F.adaptive_avg_pool2d(x, (1, 1))
        output = self.fc1(output)
        # 原文中仅说使用ReLU, pytorch官方实现代码中使用ReLU
        output = F.relu(output, inplace=True)
        output = self.fc2(output)
        # 使用hard-sigmoid激活函数
        output = F.hardsigmoid(output, inplace=True)
        return output * x


class InvertedResidualConfig(nn.Module):
    def __init__(self, input_channel: int, kernel_size: int, expand_channel: int, output_channel: int,
                 use_se: bool, activation: str, stride: int, width_mult: float):
        """
        倒置残差结构相关配置信息
        Args:
            input_channel: 输入通道数
            output_channel: 输出通道数
            kernel_size: 卷积核大小
            expand_channel: 倒置残差结构第一个1x1卷积层升维后通道数
            activation: 激活函数, HS为True
            use_se: 是否使用SE注意力机制
            stride: 步长
            width_mult: 宽度乘法器
        """
        super(InvertedResidualConfig, self).__init__()
        self.input_channel = int(input_channel * width_mult)
        self.output_channel = int(output_channel * width_mult)
        self.kernel_size = kernel_size
        self.expand_channel = int(expand_channel * width_mult)
        self.activation = activation == 'HS'
        self.use_se = use_se
        self.stride = stride


class InvertedResidual(nn.Module):
    def __init__(self, cnf: InvertedResidualConfig):
        super(InvertedResidual, self).__init__()
        # 判断传入的步长是否合法
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')
        # 判断是否符合使用快捷连接
        self.use_shortcut = cnf.stride == 1 and cnf.input_channel == cnf.output_channel
        layers = []
        activation_layer = nn.Hardswish if cnf.activation else nn.ReLU
        # 判断是否需要使用1x1卷积升维
        if cnf.input_channel != cnf.expand_channel:
            layers.append(BasicBlock(cnf.input_channel, cnf.expand_channel, activation_layer=activation_layer))

        # depth wise
        layers.append(BasicBlock(cnf.expand_channel, cnf.expand_channel, kernel_size=cnf.kernel_size,
                                 stride=cnf.stride, groups=cnf.expand_channel, activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(SqueezeExcitation(cnf.expand_channel))

        # project
        # 倒置残差结构第二个1x1卷积激活函数使用线性激活
        layers.append(BasicBlock(cnf.expand_channel, cnf.output_channel, kernel_size=1,
                                 stride=1, groups=1, activation_layer=nn.Identity))

        self.bneck = nn.Sequential(*layers)

    def forward(self, x):
        result = self.bneck(x)
        if self.use_shortcut:
            result += x
        return result


class MobileNetV3(nn.Module):
    def __init__(self, inverted_residual_setting: List[InvertedResidualConfig], num_classes=1000, last_channel=1280):
        super(MobileNetV3, self).__init__()
        if inverted_residual_setting is None:
            raise ValueError("The inverted_residual_setting should not be empty")

        block = InvertedResidual
        layers: List[nn.Module] = []

        # 构建第一个卷积层
        first_output_channel = inverted_residual_setting[0].input_channel
        layers.append(BasicBlock(3, first_output_channel, kernel_size=3, stride=2,
                                 groups=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.Hardswish))

        # 构建倒置残差结构块
        for cnf in inverted_residual_setting:
            layers.append(block(cnf))

        # 构建特征层最后一个卷积
        last_conv_input_channels = inverted_residual_setting[-1].output_channel
        # 根据论文中表格最后一个输出层时输入层6倍
        last_conv_output_channels = last_conv_input_channels * 6
        layers.append(BasicBlock(last_conv_input_channels, last_conv_output_channels,
                                 1, activation_layer=nn.Hardswish))

        self.features = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes)
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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


# def create_mobilenet_v3(arch: str, num_classes: int = 1000, width_mult: float = 1.0):
#     bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
#
#     if arch == "mobilenet_v3_large":
#         inverted_residual_setting = [
#             bneck_conf(16, 3, 16, 16, False, "RE", 1),
#             bneck_conf(16, 3, 64, 24, False, "RE", 2),
#             bneck_conf(24, 3, 72, 24, False, "RE", 1),
#             bneck_conf(24, 5, 72, 40, True, "RE", 2),
#             bneck_conf(40, 5, 120, 40, True, "RE", 1),
#             bneck_conf(40, 5, 120, 40, True, "RE", 1),
#             bneck_conf(40, 3, 240, 80, False, "HS", 2),
#             bneck_conf(80, 3, 200, 80, False, "HS", 1),
#             bneck_conf(80, 3, 184, 80, False, "HS", 1),
#             bneck_conf(80, 3, 184, 80, False, "HS", 1),
#             bneck_conf(80, 3, 480, 112, True, "HS", 1),
#             bneck_conf(112, 3, 672, 112, True, "HS", 1),
#             bneck_conf(112, 5, 672, 160, True, "HS", 2),
#             bneck_conf(160, 5, 960, 160, True, "HS", 1),
#             bneck_conf(160, 5, 960, 160, True, "HS", 1)
#         ]
#         last_channel = 1280
#     elif arch == "mobilenet_v3_small":
#         inverted_residual_setting = [
#             bneck_conf(16, 3, 16, 16, True, "RE", 2),
#             bneck_conf(16, 3, 72, 24, False, "RE", 2),
#             bneck_conf(24, 3, 88, 24, False, "RE", 1),
#             bneck_conf(24, 5, 96, 40, True, "HS", 2),
#             bneck_conf(40, 5, 240, 40, True, "HS", 1),
#             bneck_conf(40, 5, 240, 40, True, "HS", 1),
#             bneck_conf(40, 5, 120, 48, True, "HS", 1),
#             bneck_conf(48, 5, 144, 48, True, "HS", 1),
#             bneck_conf(48, 5, 288, 96, True, "HS", 2),
#             bneck_conf(96, 5, 576, 96, True, "HS", 1),
#             bneck_conf(96, 5, 576, 96, True, "HS", 1)
#         ]
#         last_channel = 1024
#     else:
#         raise ValueError("Unsupported model type {}".format(arch))
#
#     model = MobileNetV3(inverted_residual_setting, num_classes, last_channel)
#     return model


def create_v3_large(num_classes: int = 1000, width_mult: float = 1.0):
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160, True, "HS", 2),
        bneck_conf(160, 5, 960, 160, True, "HS", 1),
        bneck_conf(160, 5, 960, 160, True, "HS", 1)
    ]
    last_channel = 1280

    model = MobileNetV3(inverted_residual_setting, num_classes, last_channel)
    return model


def create_v3_small(num_classes: int = 1000, width_mult: float = 1.0):
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, True, "RE", 2),
        bneck_conf(16, 3, 72, 24, False, "RE", 2),
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96, True, "HS", 2),
        bneck_conf(96, 5, 576, 96, True, "HS", 1),
        bneck_conf(96, 5, 576, 96, True, "HS", 1)
    ]
    last_channel = 1024
    model = MobileNetV3(inverted_residual_setting, num_classes, last_channel)
    return model
