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
BATCH_SIZE = 32
LR = 1e-2
lr_decay_step = 1 

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据准备
# ImageNet数据集上训练出来的正则化均值和方差
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256)),                                             # Resize((256))会等比例将最短边缩放到256, Resize((256, 256))将两边同时缩放到256
    transforms.CenterCrop(256),                                           # 中心裁剪，裁剪大小为256
    transforms.RandomCrop(224),                                           # 随机裁剪，裁剪大小为224
    transforms.RandomHorizontalFlip(p=0.5),                               # 随机水平翻转
    transforms.ToTensor(),                                                # 转换成Tensor数据类型
    transforms.Normalize(norm_mean, norm_std)                             # 正则化
])

normalizes = transforms.Normalize(norm_mean, norm_std)

valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),                                    # 将图像转换成(256, 256)大小
        transforms.TenCrop(224, vertical_flip=False),                     # 将图像四个角和中心裁剪包括水平翻转后，一共裁剪十个块，不包括垂直翻转，大小为224×224
        transforms.Lambda(lambda crops: torch.stack([normalizes(transforms.ToTensor()(crop)) for crop in crops])), 
])

# 构建Dataset
train_dataset = MyDataset(root=data_dir, train=True, transform=train_transform)
valid_dataset = MyDataset(root=data_dir, train=False, transform=valid_transform)

# 构建DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=4)

# 数据集长度
train_size = len(train_dataset)
valid_size = len(valid_dataset)

# 构建Model
model = MyVGG()
model.to(device)

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=LR,momentum=0.9)

# 设置学习率下降策略
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)

# 模型训练
total_train_step = 0
total_test_step = 0
epoch = 10

# 创建tensorboard
writer = SummaryWriter('./logs')

for i in range(epoch):
    print("=================第 {} 轮 训练开始=================".format(i + 1))
    model.train(True)
    for data in train_dataloader:
        start_time = time.time()
        # 前向传播
        imgs, targets = data
        imgs, targets = imgs.to(device),targets.to(device)
        outputs = model(imgs)

        # 计算训练误差
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录训练次数
        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(
                "训练时间为: {}, 训练次数为: {}, Loss: {}".format(end_time - start_time, total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss)
    
    # 更新学习率
    scheduler.step()
    
    # 模型测试
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in valid_dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device),targets.to(device)

            bs, ncrops, c, h, w = imgs.size()
            outputs = model(imgs.view(-1, c, h, w))
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

            loss = loss_fn(outputs_avg, targets)
            total_test_loss += loss.item()
            accuracy = (outputs_avg.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / valid_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / valid_size, total_test_step)

    # 保存模型
    torch.save(model.state_dict(), "./model/model_{}.pth".format(i))
    print("模型{}保存成功".format(i))

writer.close()