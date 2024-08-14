import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List
from functools import partial


class SSEBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int):
        # SSE模块一个分支只进行BatchNorm, 输入特征矩阵形状不会改变
        # SSE另一个分支会先进行全局平均池化, 然后使用1x1卷积, 其中输出通道数仅受1x1卷积影响
        # SSE模块最好需要将两个分支的输出结果进行相乘再输出, 为保证输出结果相同, 1x1卷积的输入通道数应当等于输出通道数
        super(SSEBlock, self).__init__()
        self.norm = nn.BatchNorm2d(in_planes)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        bn = self.norm(x)
        output = torch.sigmoid(self.fc(self.pool(x)))

        return torch.mul(bn, output)


class DownSampling(nn.Module):
    def __init__(self, in_planes: int, out_planes: int):
        super(DownSampling, self).__init__()
        self.branch1 = nn.Sequential(
            # 图像长宽减半, AvgPool2d中padding默认等于kernel_size
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        self.branch1_last = nn.Sequential(
            # 最后一个downsampling层, 输入图像为7x7, 为了使得与branch2输出图像尺寸相同, 设置padding=1
            nn.AvgPool2d(kernel_size=2, padding=1),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        self.branch2 = nn.Sequential(
            # 输入图像长宽减半, padding默认为0
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(out_planes)
        )
        self.branch3 = nn.Sequential(
            # 图像从[b, c, h, w] -> [b, c, 1, 1]
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.data.size()
        if h % 2 == 1:
            branch1 = self.branch1_last(x)
        else:
            branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        result = branch1 + branch2
        result = result * branch3
        return F.silu(result, inplace=True)


class FusionBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int):
        super(FusionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        # 由于在最开始有一个concatenation, 导致Fusion模块内通道数翻倍
        self.groups = in_planes
        self.downsample = DownSampling(in_planes * 2, out_planes)

    # 通道混洗
    def channel_shuffle(self, x):
        b, c, h, w = x.data.size()
        channels_per_groups = c // self.groups

        x = x.reshape(b, channels_per_groups, self.groups, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, c, h, w)

        return x

    def forward(self, input1, input2):
        # concatenation
        x = torch.cat([self.bn(input1), self.bn(input2)], dim=1)
        # 通道混洗
        x = self.downsample(x)

        return x


class FuseBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int):
        super(FuseBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        result = branch1 + branch2
        return result


class Stream(nn.Module):
    def __init__(self, in_planes: int, out_planes: int):
        super(Stream, self).__init__()
        self.fuse = FuseBlock(in_planes, out_planes)
        self.sse = SSEBlock(in_planes, out_planes)

    def forward(self, x):
        fuse = self.fuse(x)
        sse = self.sse(x)
        result = F.silu(fuse + sse, inplace=True)
        return result


class ParNet(nn.Module):
    def __init__(self, ds_channels: List[int], stream_nums: List[int], num_classes: int):
        """
        Args:
            ds_channels: 列表类型, 存放除stream1分支上的下采样层外的所有下采样层输出通道数
            stream_nums: 各个stream流中RepVGG SSE Block数
            num_classes: 分类数量
        """
        super(ParNet, self).__init__()
        self.ds1 = DownSampling(3, ds_channels[0])
        self.ds2 = DownSampling(ds_channels[0], ds_channels[1])
        self.ds3 = DownSampling(ds_channels[1], ds_channels[2])
        self.ds4 = DownSampling(ds_channels[2], ds_channels[3])
        self.ds5 = DownSampling(ds_channels[3], ds_channels[4])

        self.stream1 = nn.Sequential(
            *[Stream(ds_channels[1], ds_channels[1]) for _ in range(stream_nums[0])]
        )
        # 下采样层, 输出通道数等于stream2层输出通道数, 然后进行Fusion操作
        self.stream1_downsample = DownSampling(ds_channels[1], ds_channels[2])

        self.stream2 = nn.Sequential(
            *[Stream(ds_channels[2], ds_channels[2]) for _ in range(stream_nums[1])]
        )
        # 将stream1和stream2进行融合, 输出通道数为stream3层输出通道数, 以便进行融合
        self.stream2_fusion = FusionBlock(ds_channels[2], ds_channels[3])

        self.stream3 = nn.Sequential(
            *[Stream(ds_channels[3], ds_channels[3]) for _ in range(stream_nums[2])]
        )
        self.stream3_fusion = FusionBlock(ds_channels[3], ds_channels[3])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=ds_channels[4], out_features=num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 连续两个下采样层
        # ParNet-S
        # N x 3 x 224 x 224
        x = self.ds1(x)
        # N x 64 x 112 x 112
        x = self.ds2(x)
        # N x 96 x 56 x 56

        # 将第二个下采样结果传递给stream1
        stream1 = self.stream1(x)
        # N x 96 x 56 x 56
        # stream1经过一连串RepVGG SSE Block后进入stream1-downsampling
        stream1 = self.stream1_downsample(stream1)
        # N x 192 x 28 x 28

        # 进行第三个下采样
        x = self.ds3(x)
        # N x 192 x 28 x 28
        # 第三个下采样结果传递给stream2
        stream2 = self.stream2(x)
        # N x 192 x 28 x 28
        # 将stream1和stream2结果融合
        stream2 = self.stream2_fusion(stream1, stream2)
        # N x 384 x 14 x 14

        # 进行第四个下采样
        x = self.ds4(x)
        # N x 384 x 14 x 14
        # 第四个下采样结果传递给stream3
        stream3 = self.stream3(x)
        # N x 384 x 14 x 14
        # 将stream2和stream3结果进行融合
        stream3 = self.stream3_fusion(stream2, stream3)
        # N x 384 x 7 x 7

        # 上述结果融合过后进行最后一个下采样
        x = self.ds5(stream3)
        # N x 1280 x 7 x 7

        # 分类器
        result = self.classifier(x)
        return result


def create_model(model_name: str, num_classes: int):
    stream_nums = [4, 5, 5]
    if model_name == "ParNet-S":
        ds_channels = [64, 96, 192, 384, 1280]
    elif model_name == "ParNet-M":
        ds_channels = [64, 128, 256, 512, 2048]
    elif model_name == "ParNet-L":
        ds_channels = [64, 160, 320, 640, 2560]
    elif model_name == "ParNet-XL":
        ds_channels = [64, 200, 400, 800, 3200]
    else:
        raise ValueError(f"Model name {model_name} not supported")
    model = ParNet(ds_channels, stream_nums, num_classes)

    return model
