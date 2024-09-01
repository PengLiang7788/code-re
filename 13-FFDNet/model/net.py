import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary


class FFDNet(nn.Module):
    def __init__(self, img_channels=1):
        """
        Args:
            img_channels: 输入图像通道数 1-灰度图 3-彩色图
        """
        super(FFDNet, self).__init__()
        if img_channels == 1:
            # 灰度图
            depth = 15
            middle_channels = 64
        elif img_channels == 3:
            # 彩色图
            depth = 12
            middle_channels = 96
        else:
            raise ValueError('img_channels must be 1 or 3')
        sca = 2
        output_channels = img_channels * sca * sca

        # 下采样
        self.down_sample = nn.PixelUnshuffle(downscale_factor=sca)

        # 中间DnCNN结构
        layers = nn.ModuleList()
        # 第一层 Conv + ReLU
        layers.append(nn.Conv2d(img_channels * sca * sca + 1, middle_channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))

        # 中间层 Conv + BN + ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(middle_channels, momentum=0.9, eps=1e-4, affine=True))
            layers.append(nn.ReLU(inplace=True))

        # 最后层, Conv
        layers.append(nn.Conv2d(middle_channels, output_channels, kernel_size=3, padding=1, bias=True))
        self.model = nn.Sequential(*layers)

        # 上采样
        self.upsample = nn.PixelShuffle(upscale_factor=sca)

    def forward(self, x, sigma):
        # 对输入图像进行填充, 确保高宽为偶数
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 2) * 2 - h)
        paddingRight = int(np.ceil(w / 2) * 2 - w)
        x = nn.ReplicationPad2d((0, paddingBottom, 0, paddingRight))(x)

        # 下采样
        x = self.down_sample(x)
        # 生成noise level map
        m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        x = torch.cat((x, m), dim=1)
        # 进入CNN
        x = self.model(x)
        # 上采样
        x = self.upsample(x)
        # 恢复成原始形状
        x = x[..., :h, :w]
        return x


if __name__ == '__main__':
    model = FFDNet(1)
    # summary(model, input_size=(2, 1, 240, 240))
    x = torch.randn((2, 1, 240, 240))
    sigma = torch.randn(2, 1, 1, 1)
    x = model(x, sigma)
    print(x.shape)
