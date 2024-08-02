import argparse
import os
import time 
from datetime import datetime
import shutil

import torch
import torch.nn as nn
import random
import numpy as np
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from utils import AverageMeter, accuracy
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from model import densenet

parser = argparse.ArgumentParser(description="PyTorch DenseNeck Training")

parser.add_argument('--epoch', type=int, default=100, help="number of total epochs to train")
parser.add_argument('--start_epoch', default=0, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument('--batch_size', type=int, default=64, help="mini-batch size (default: 64)")
parser.add_argument('--lr', '--learning_rate', type=float, default=0.1, help="initial learning rate")
parser.add_argument('--momentum', default=0.9, type=int, help="momentum")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay (default: 1e-4)")
parser.add_argument('--print_freq', '-p', default=1, type=int, help="print frequency (default: 10)")
parser.add_argument('--layers', default=100, type=int, help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int, help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')
# 命令行参数 no-augmet, 解析对象时使用augment, 默认为True, 当在命令行中使用了no-augment参数时则为False
parser.add_argument('--no-augment', dest='augment', action='store_false', help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float, help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false', help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet_BC_100_12', type=str, help='name of experiment')
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', action='store_true')
parser.add_argument("--seed", type=int, default=33, help="random seed")
parser.add_argument('--workers', type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()

def seed_torch(seed=123):
    # 设置随机数生成器的种子，确保实验可重复性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_torch(seed=args.seed)

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置实验结果保存路径
# exp/modelName_datetime_lr_wd
lr = args.lr
weight_decay = args.weight_decay
# 将weight_decay转换为科学计数法
weight_decay = format(weight_decay, '.1e')
# 获取当前日期
now = datetime.now().strftime(r'%mM_%dD_%HH')
exp_path = "./exp/{}_{}_lr{}_wd{}".format(args.name, now, lr, weight_decay)
os.makedirs(exp_path, exist_ok=True)
# exp_path/runs: tensorboard保存的文件;
# exp_path/ckpt: 权重文件

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

if args.augment:
    train_trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
else:
    train_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

test_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# 创建数据集
train_set = torchvision.datasets.CIFAR10(root="./data", train=True, transform=train_trans, download=True)
val_set = torchvision.datasets.CIFAR10(root="./data", train=False, transform=test_trans, download=True)

# 创建数据加载器
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                          num_workers=args.workers, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, 
                        num_workers=args.workers, pin_memory=True)

# train
def train_one_epoch(model:nn.Module, optimizer:torch.optim, train_loader:DataLoader, loss_fn:nn.CrossEntropyLoss):
    model.train()
    acc_recoder = AverageMeter()  # 用于计算精度的工具
    loss_recoder = AverageMeter() # 用于计算损失的工具

    for (imgs, targets) in tqdm(train_loader, desc="train"):
        imgs, targets = imgs.to(device), targets.to(device)   # 将数据集放在相应设备上
        outputs = model(imgs)                                 # 模型预测
        loss = loss_fn(outputs, targets)                      # 损失函数计算损失
        loss_recoder.update(loss.item(), n=imgs.size(0))      # 记录损失
        acc = accuracy(outputs, targets)[0]                   # 计算精度
        acc_recoder.update(acc.item(), n=imgs.size(0))        # 记录精度
        optimizer.zero_grad()                                 # 梯度清零
        loss.backward()                                       # 反向传播
        optimizer.step()                                      # 更新模型参数
    
    losses = loss_recoder.avg   # 计算平均损失
    acces = acc_recoder.avg     # 计算平均精度

    return acces, losses

def evaluation(model: nn.Module, val_loader: DataLoader, loss_fn: nn.CrossEntropyLoss):
    model.eval()
    acc_recoder = AverageMeter()
    loss_recoder = AverageMeter()

    with torch.no_grad():
        for (imgs, labels) in tqdm(val_loader, desc="Evaluation"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            acc = accuracy(outputs, labels)[0]
            loss = loss_fn(outputs, labels)
            acc_recoder.update(acc.item(), imgs.size(0))
            loss_recoder.update(loss.item(), imgs.size(0))
    
    losses = loss_recoder.avg
    acces = acc_recoder.avg

    return losses, acces

def train(model:nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
          optimizer: torch.optim, scheduler: CosineAnnealingLR, loss_fn: nn.CrossEntropyLoss):
    # 记录训练开始时间
    start_time = time.time()
    # 初始最佳准确率为-1, 以便保存最佳模型
    best_acc = -1
    # 打开一个用于写入训练过程的信息文件
    f = open(os.path.join(exp_path,"lr{}_wd{}.txt".format(lr, weight_decay)), "w")

    for epoch in range(args.start_epoch, args.epoch):
        # 在训练集上执行一个周期训练
        train_losses, train_access = train_one_epoch(model, optimizer, train_loader, loss_fn)
        # 在测试集上评估模型性能，获取测试损失和准确率
        val_losses, val_access = evaluation(model, val_loader, loss_fn)
        # 如果当前测试准确率高于历史最佳准确率，更新最佳准确率，并保存模型参数
        if val_access > best_acc:
            best_acc = val_access
            state_dict = dict(epoch=epoch+1, model=model.state_dict(), acc=val_access)
            name = os.path.join(exp_path, "ckpt", "best.pth")
            os.makedirs(os.path.dirname(name), exist_ok=True)
            torch.save(state_dict, name)
        
        # 更新学习率
        scheduler.step()
        # 定义要记录的训练信息标签
        tags = ['train_losses', 'train_access', 'val_losses', 'val_access']
        tb_writer.add_scalar(tags[0], train_losses, epoch + 1)
        tb_writer.add_scalar(tags[1], train_access, epoch + 1)
        tb_writer.add_scalar(tags[2], val_losses, epoch + 1)
        tb_writer.add_scalar(tags[3], val_access, epoch + 1)

        # 打印训练过程信息，以及将信息写入文件
        if (epoch + 1) % args.print_freq == 0: #  print_freq指定为1 则每轮都打印
            msg = "epoch:{} model:{} train loss:{:.2f} acc:{:.2f}  test loss{:.2f} acc:{:.2f}\n".format(
                epoch + 1,
                args.name,
                train_losses,
                train_access,
                val_losses,
                val_access,
            )
            print(msg)
            f.write(msg)
            f.flush()
    # 输出训练结束后的最佳准确度和总训练时间
    msg_best = "model:{} best acc:{:.2f}\n".format(args.name, best_acc)
    time_elapsed = "traninng time: {}".format(time.time() - start_time)
    print(msg_best)
    f.write(msg_best)
    f.write(time_elapsed)
    f.close()

if __name__ == '__main__':
    tb_path = os.path.join(exp_path, "runs")
    tb_writer = SummaryWriter(log_dir=tb_path)

    model = densenet.DenseNet(args.layers, 10, args.growth, reduction=args.reduce,
                              bottleNeck=args.bottleneck, drop_rate=args.droprate)
    model.to(device)
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, 
                                weight_decay=args.weight_decay, nesterov=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)
    train(model, train_loader, val_loader, optimizer, scheduler, loss_fn)
    tb_writer.close()
