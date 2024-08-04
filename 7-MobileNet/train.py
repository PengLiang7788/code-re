import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import random
from datetime import datetime
import time
from datasets import MyDataset
from utils import AverageMeter, accuracy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model.net import MobileNet

parser = argparse.ArgumentParser("Pytorch MobileNet Training")
parser.add_argument('--epochs', type=int, default=100, help="number of total epochs to train")
parser.add_argument('--batch_size', type=int, default=64, help="mini-batch size (default: 64)")
parser.add_argument('--lr', '--learning_rate', type=float, default=0.1, help="initial learning rate")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay (default: 1e-4)")
parser.add_argument('--print_freq', '-p', default=1, type=int, help="print frequency (default: 1)")
parser.add_argument('--name', default='MobileNet_v1', type=str, help='name of experiment')
parser.add_argument("--seed", type=int, default=33, help="random seed")
parser.add_argument('--workers', type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
parser.add_argument('--split_ratio', type=float, default=0.8, help="split ratio")

args = parser.parse_args()


def seed_torch(seed=123):
    # 设置随机数生成器的种子，确保实验可重复性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 设置随机数种子, 确保实验可重复性
seed_torch(args.seed)

data_dir = "../data/train"

# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置实验结果保存路径
# exp/modelName_datetime_lr_wd
model_name = args.name
lr = args.lr
weight_decay = args.weight_decay
wd = format(weight_decay, '.1e')
now = datetime.now().strftime(r"%mM_%dD_%HH")
exp_path = "exp/{}_{}_lr{}_wd{}".format(now, model_name, lr, weight_decay)
os.makedirs(exp_path, exist_ok=True)

# 定义数据转换
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(norm_mean, norm_std)

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize(256),  # 将短边缩放到256
        transforms.CenterCrop(256),  # 中心裁剪
        transforms.RandomCrop(224),  # 随机裁剪224x224大小的图像
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.ToTensor(),  # 转为Tensor数据类型
        transforms.Normalize(norm_mean, norm_std)  # 正则化
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.TenCrop(224, vertical_flip=False),
        transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))
    ])
}

train_set = MyDataset(data_dir, mode="train", transform=data_transforms["train"],
                      split_ratio=args.split_ratio, rng_seed=args.seed)
val_set = MyDataset(data_dir, mode="val", transform=data_transforms["val"],
                    split_ratio=args.split_ratio, rng_seed=args.seed)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True)


# 训练一轮
def train_one_epoch(model: nn.Module, optimizer: optim, train_loader: DataLoader, loss_fn: nn.CrossEntropyLoss):
    model.train(True)
    acc_recoder = AverageMeter()  # 记录准确率
    loss_recoder = AverageMeter()  # 记录损失

    for imgs, labels in tqdm(train_loader, desc="train"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        loss_recoder.update(loss.item(), imgs.size(0))
        acc = accuracy(outputs, labels)[0]
        acc_recoder.update(acc.item(), imgs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses = loss_recoder.avg
    acc = acc_recoder.avg

    return losses, acc


def evaluation(model: nn.Module, val_loader: DataLoader, loss_fn: nn.CrossEntropyLoss):
    model.eval()
    acc_recoder = AverageMeter()
    loss_recoder = AverageMeter()

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Evaluation"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            loss_recoder.update(loss.item(), imgs.size(0))
            acc = accuracy(outputs, labels)[0]
            acc_recoder.update(acc.item(), imgs.size(0))

    losses = loss_recoder.avg
    acc = acc_recoder.avg

    return losses, acc


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          optimizer: optim, loss_fn: nn.CrossEntropyLoss, scheduler: optim.lr_scheduler.CosineAnnealingLR):
    # 记录训练开始时间
    start_time = time.time()
    # 记录最佳准确率, 初始为-1
    best_acc = -1
    # 打开一个用于写入训练过程的信息文件
    f = open(os.path.join(exp_path, "lr{}_wd{}.txt".format(lr, weight_decay)), "w")

    for epoch in range(args.epochs):
        train_losses, train_acc = train_one_epoch(model, optimizer, train_loader, loss_fn)
        val_losses, val_acc = evaluation(model, val_loader, loss_fn)

        if val_acc > best_acc:
            best_acc = val_acc
            state_dict = dict(epoch=epoch + 1, model=model.state_dict(), acc=val_acc)
            pth_path = os.path.join(exp_path, "ckpt", "best.pth")
            os.makedirs(pth_path, exist_ok=True)
            torch.save(state_dict, pth_path)

        # 更新学习率
        scheduler.step()
        # 定义要记录的训练信息标签
        tags = ['train_losses', 'train_access', 'val_losses', 'val_access']
        tb_writer.add_scalar(tags[0], train_losses, epoch + 1)
        tb_writer.add_scalar(tags[1], train_acc, epoch + 1)
        tb_writer.add_scalar(tags[2], val_losses, epoch + 1)
        tb_writer.add_scalar(tags[3], val_acc, epoch + 1)

        # 打印训练过程信息，以及将信息写入文件
        if (epoch + 1) % args.print_freq == 0:  # print_freq指定为1 则每轮都打印
            msg = "epoch:{} model:{} train loss:{:.2f} acc:{:.2f}  test loss{:.2f} acc:{:.2f}\n".format(
                epoch + 1,
                args.name,
                train_losses,
                train_acc,
                val_losses,
                val_acc,
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
    os.makedirs(tb_path, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_path)

    model = MobileNet(num_classes=2)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train(model, train_loader, val_loader, optimizer, loss_fn, scheduler)
    tb_writer.close()
