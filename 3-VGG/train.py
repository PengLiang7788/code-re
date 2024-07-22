import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from datasets import MyDataset
from model import MyVGG
import torchvision
from torchvision import transforms

data_dir = '../data/train'
log_dir = './logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取数据
# ImageNet数据集上训练出来的正则化均值和方差
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256)),                 # Resize((256))会等比例将最短边缩放到256, Resize((256, 256))将两边同时缩放到256
    transforms.CenterCrop(256),               # 中心裁剪，裁剪大小为256
    transforms.RandomCrop(224),               # 随机裁剪，裁剪大小为224
    transforms.RandomHorizontalFlip(p=0.5),   # 随机水平翻转
    transforms.ToTensor(),                    # 转换成Tensor数据类型
    transforms.Normalize(norm_mean, norm_std) # 正则化
])

valid_transform = transforms.Compose([
    
])
train_dataset = MyDataset(root=data_dir, train=True, transform=train_transform)


