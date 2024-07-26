import torch
import torch.nn as nn
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=False, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.conv1 = BasicConv2d(3, 64, kernel_size =7, stride=2, padding=2)
        # ceil_mode=True向上取整
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()


class BasicConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        return self.relu(x)

if __name__ == "__main__":
    model = GoogLeNet()
    print(model)

