import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchsummary import summary


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
        # 第一层使用了偏置, 中间层没使用, 默认bias为True
        # 一般是跟BN的Conv的bias为False, 因为对BN的计算没用, 可以减少计算量. 如果不考虑计算量, 加不加都行
        layers.append(nn.Conv2d(img_channels, features, kernel_size=kernel_size, padding=padding, stride=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # 中间层 Conv+BN+ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=kernel_size, padding=padding, stride=1, bias=False))
            # eps避免计算计算标准差分母为0, 一般为1e-5(默认) 或 1e-4
            # momentum更新率, 一般为0.9或0.95; 默认为0.1, 即只使用新批次的10%数据, 之前的90%来自移动平均值
            # 切块的数据形式, BN动量接近1会比较好
            layers.append(nn.BatchNorm2d(features, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))

        # 最后一层 Conv
        layers.append(nn.Conv2d(features, img_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        # DnCNN 是从图像中得出噪声v
        out = self.dncnn(x)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = DnCNN()

    summary(model, (1, 180, 180))