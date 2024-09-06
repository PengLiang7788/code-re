import torch
import torch.nn as nn


class Downsample(nn.Module):
    """
    对应原文的下采样层
    """
    def __init__(self, in_planes, nd, kd, padding=1, stride=2):
        super(Downsample, self).__init__()
        self.padder = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_planes, nd, kernel_size=kd, stride=stride)
        self.bn1 = nn.BatchNorm2d(nd)

        self.conv2 = nn.Conv2d(in_channels=nd, out_channels=nd, kernel_size=kd, stride=1)
        self.bn2 = nn.BatchNorm2d(nd)

        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.padder(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x



