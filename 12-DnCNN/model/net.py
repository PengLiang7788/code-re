import torch
import torch.nn as nn
import torch.nn.functional as F


class DnCNN(nn.Module):
    def __init__(self, depth=17, features=64, img_channels=1, kernel_size=3):
        """
        Args:
            depth: 网络深度, 默认深度为17
            features: 中间层通道数
            img_channels: 输入图像通道数 灰度图为1 彩色图为3
            kernel_size: 卷积核大小, 默认为3
        """
        super(DnCNN, self).__init__()
        layers = nn.ModuleList()
        # 使中间特征图尺寸与输入图像尺寸保持相同
        padding = 1

        # 第一层 Conv+ReLU
        layers.append(nn.Conv2d(img_channels, features, kernel_size=kernel_size, padding=padding, stride=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # 中间层 Conv+BN+ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=kernel_size, padding=padding, stride=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        # 最后一层 Conv
        layers.append(nn.Conv2d(features, img_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
