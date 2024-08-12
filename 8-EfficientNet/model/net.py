import copy
import math

import torch
import torch.nn as nn
from typing import Optional, Callable, List
from torch.nn import functional as F
from collections import OrderedDict
from functools import partial


def _make_divisible(ch, divisor=8, min_ch=None):
    # 将通道数调整到最近的8的倍数
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNActivation(nn.Sequential):
    # conv - BN - activation
    def __init__(self, in_planes: int, out_planes: int,
                 kernel_size: int = 3, stride: int = 1, groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU

        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, in_planes: int, reduction: int = 4):
        super(SqueezeExcitation, self).__init__()
        # 第一个卷积层 将channel下降到原来的 1/reduction
        squeeze_channel = in_planes // reduction
        self.fc1 = nn.Conv2d(in_planes, squeeze_channel, kernel_size=1, bias=False)
        # 第二个卷积层 将channel升维到输入维度
        self.fc2 = nn.Conv2d(squeeze_channel, in_planes, kernel_size=1, bias=False)

    def forward(self, x):
        output = F.adaptive_avg_pool2d(x, (1, 1))
        output = self.fc1(output)
        output = F.silu(output, inplace=True)
        output = self.fc2(output)
        output = F.silu(output, inplace=True)
        return output * x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MBConvConfig:
    def __init__(self, kernel_size: int, in_planes: int, out_planes: int,
                 expand_ratio: int, stride: int, use_se: bool,
                 drop_rate: float, index: str, width_coefficient: float):
        """
        Args:
            kernel_size: MBConv中卷积核大小 3 or 5
            in_planes: MBConv中输入特征图通道数
            out_planes: MBConv输出特征图通道数
            expand_ratio: MBConv第一个特征图升维比例
            stride: 深度卷积部分步长
            use_se: 是否使用SE注意力机制
            drop_rate: MBConv中的Dropout层随机失活比率
            index: 记录当前MBConv模块名称
            width_coefficient: 宽度倍率因子
        """
        self.kernel_size = kernel_size
        self.in_planes = self.adjust_channels(in_planes, width_coefficient)
        self.out_planes = self.adjust_channels(out_planes, width_coefficient)
        self.expand_planes = self.in_planes * expand_ratio
        self.stride = stride
        self.use_se = use_se
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        # 将channels * 宽度倍率因子, 再调整到8的整数倍
        return _make_divisible(channels * width_coefficient, 8)


class MBConv(nn.Module):
    def __init__(self, cnf: MBConvConfig, norm_layer: Optional[Callable[..., nn.Module]]):
        """
        Args:
            cnf: MBConv层配置文件
            norm_layer: 正则化层
        """
        super(MBConv, self).__init__()
        if cnf.stride not in [1, 2]:
            raise ValueError('Stride must be 1 or 2.')
        # 只有当步长为1且输入通道数等于输出通道数才使用快捷连接
        self.short_cut = cnf.stride == 1 and cnf.in_planes == cnf.out_planes
        layers = OrderedDict()
        activation_layer = nn.SiLU

        # 第一个1x1卷积层 升维
        # 当expand_ratio = 1时, expand_planes = in_planes, 不需要进行升维
        if cnf.expand_planes != cnf.in_planes:
            layers.update({"expand_conv": ConvBNActivation(cnf.in_planes, cnf.expand_planes,
                                                           kernel_size=1, stride=1, norm_layer=norm_layer,
                                                           activation_layer=activation_layer, groups=1)})
        # 深度卷积层
        layers.update({"dwconv": ConvBNActivation(cnf.expand_planes, cnf.expand_planes, kernel_size=cnf.kernel_size,
                                                  stride=cnf.stride, groups=cnf.expand_planes, norm_layer=norm_layer,
                                                  activation_layer=activation_layer)})
        # 是否使用SE模块
        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.expand_planes)})

        # 第二个1x1卷积特征映射
        layers.update({"project_conv": ConvBNActivation(cnf.expand_planes, cnf.out_planes, kernel_size=1,
                                                        norm_layer=norm_layer, activation_layer=activation_layer)})

        self.block = nn.Sequential(layers)

        # 只有在使用short_cut连接时才使用dropout层
        if self.short_cut and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        output = self.block(x)
        output = self.dropout(output)

        if self.short_cut:
            output += x
        return output


class EfficientNet(nn.Module):
    def __init__(self, width_coefficient: float, depth_coefficient: float, num_classes: int = 1000,
                 dropout_rate: float = 0.2, drop_connect_rate: float = 0.2,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        """
        Args:
            width_coefficient: 宽度倍率因子, 论文中为w
            depth_coefficient: 深度倍率因子, 论文中d
            num_classes: 分类的类别个数
            dropout_rate: stage9 FC层前面的Dropout随即失活比率
            drop_connect_rate: MBConv模块中Dropout层随机失活比率
            block: MBConv模块
            norm_layer: 正则化层
        """
        super(EfficientNet, self).__init__()

        # 默认B0网络配置文件, B1-B7都是在此基础上更改宽度倍率因子和深度倍率因子
        # kernel_size, in_channels, out_channels, exp_ratio, strides, use_se, drop_connect_rate, repeats
        default_config = [[3, 32, 16, 1, 1, True, dropout_rate, 1],
                          [3, 16, 24, 6, 2, True, dropout_rate, 2],
                          [5, 24, 40, 6, 2, True, dropout_rate, 2],
                          [3, 40, 80, 6, 2, True, dropout_rate, 3],
                          [5, 80, 112, 6, 1, True, dropout_rate, 3],
                          [5, 112, 192, 6, 2, True, dropout_rate, 4],
                          [3, 192, 320, 6, 1, True, dropout_rate, 1]]

        def round_repeats(repeats):
            # depth_coefficient: 深度倍率因子(仅针对stage2到stage8)
            # 使用depth_coefficient动态调整网络深度
            return int(math.ceil(depth_coefficient * repeats))

        # 调整网络宽度, 调整到8的整数倍
        adjust_channels = partial(MBConvConfig.adjust_channels, width_coefficient=width_coefficient)

        if block is None:
            block = MBConv
        if norm_layer is None:
            # partial方法搭建层结构, 下次使用就不需要传入eps和momentum参数
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        MB_config = partial(MBConvConfig, width_coefficient=width_coefficient)
        b = 0
        # 统计所有MBConv模块数
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_config))
        MBConv_configs: List[MBConvConfig] = []
        for stage, args in enumerate(default_config):
            cnf = copy.copy(args)
            # 搭建各个阶段的各个层
            for i in range(round_repeats(cnf.pop(-1))):
                if i > 0:
                    # 如果不是当前阶段的第一层, 那么步长为1
                    # 上面循环已经弹出了最后一个元素, 因此stride对应在倒数第三个元素
                    cnf[-3] = 1
                    # 输入通道数等于输出通道数
                    cnf[1] = cnf[2]
                cnf[-1] = args[-2] * b / num_blocks
                # stage1a, stage2a, stage2b...
                index = str(stage + 1) + chr(i + 97)
                MBConv_configs.append(MB_config(*cnf, index))
                b += 1

        layers = OrderedDict()

        # stage1
        layers.update({"stem_con": ConvBNActivation(in_planes=3, out_planes=adjust_channels(32), kernel_size=3,
                                                    stride=2, norm_layer=norm_layer)})

        # stage2-stage8 即 MBConv部分
        for cnf in MBConv_configs:
            layers.update({cnf.index: block(cnf, norm_layer)})

        # stage9
        last_conv_input_c = MBConv_configs[-1].out_planes
        last_conv_output_c = adjust_channels(1280)
        layers.update({"top": ConvBNActivation(last_conv_input_c, last_conv_output_c,
                                               kernel_size=1, norm_layer=norm_layer)})

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
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


def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b1(num_classes=1000):
    # input image size 240x240
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b2(num_classes=1000):
    # input image size 260x260
    return EfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b3(num_classes=1000):
    # input image size 300x300
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b4(num_classes=1000):
    # input image size 380x380
    return EfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b5(num_classes=1000):
    # input image size 456x456
    return EfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b6(num_classes=1000):
    # input image size 528x528
    return EfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        dropout_rate=0.5,
                        num_classes=num_classes)


def efficientnet_b7(num_classes=1000):
    # input image size 600x600
    return EfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes)
