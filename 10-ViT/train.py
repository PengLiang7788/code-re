import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
import random
import numpy as np
from datetime import datetime
import os
from my_dataset import MyDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import math
from model import create_model
from utils import AverageMeter, accuracy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time


def parse_opt():
    parser = argparse.ArgumentParser("Pytorch ViT Training")
    parser.add_argument('--epochs', type=int, default=300, help="number of total epochs to train")
    parser.add_argument('--batch_size', type=int, default=128, help="mini-batch size (default: 128)")
    parser.add_argument('--weight_decay', type=float, default=5e-5, help="weight decay (default: 5e-5)")
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--print_freq', '-p', default=1, type=int, help="print frequency (default: 1)")
    parser.add_argument('--model_name', default='vit_base_patch16_224', type=str, help='name of experiment')
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument('--workers', type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument('--split_ratio', type=float, default=0.1, help="split ratio")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--data_dir", type=str, default=r"../flower_photos", help="path to dataset")
    parser.add_argument("--num_classes", type=int, default=5, help="number of classes")
    parser.add_argument('--lrf', type=float, default=0.01)

    return parser.parse_args()


def seed_torch(seed=123):
    # 设置随机数生成器的种子，确保实验可重复性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model: nn.Module, optimizer: optim, train_loader: DataLoader, loss_fn: nn.CrossEntropyLoss):
    model.train(True)
    acc_recoder = AverageMeter()
    loss_recoder = AverageMeter()

    for imgs, lebels in tqdm(train_loader, desc="train"):
        imgs, labels = imgs.to(device), lebels.to(device)

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
          optimizer: optim, loss_fn: nn.CrossEntropyLoss, scheduler: lr_scheduler):
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
            os.makedirs(os.path.dirname(pth_path), exist_ok=True)
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
                args.model_name,
                train_losses,
                train_acc,
                val_losses,
                val_acc,
            )
            print(msg)
            f.write(msg)
            f.flush()
    # 输出训练结束后的最佳准确度和总训练时间
    msg_best = "model:{} best acc:{:.2f}\n".format(args.model_name, best_acc)
    time_elapsed = "training time: {}".format(time.time() - start_time)
    print(msg_best)
    f.write(msg_best)
    f.write(time_elapsed)
    f.close()


if __name__ == '__main__':
    args = parse_opt()
    # 设置随机数生成器种子, 确保实验的可重复性
    seed_torch(args.seed)
    # 获取数据集存放路径
    data_dir = args.data_dir
    # 设置训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置实验结果保存路径
    # exp/modelName_datetime_lr_wd
    model_name = args.model_name
    lr = args.lr
    weight_decay = args.weight_decay
    wd = format(weight_decay, ".1e")
    now = datetime.now().strftime(r"%mM_%dD_%HH")
    exp_path = "exp/{}_{}_lr{}_wd{}".format(model_name, now, lr, wd)
    os.makedirs(exp_path, exist_ok=True)

    # 创建tensorboard
    tb_path = os.path.join(exp_path, "runs")
    os.makedirs(tb_path, exist_ok=True)
    tb_writer = SummaryWriter(tb_path)

    # 定义数据转换
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
    }
    # 实例化训练数据集
    train_set = MyDataset(root_dir=data_dir, mode="train", transform=data_transforms["train"],
                          split_ratio=args.split_ratio, rng_seed=args.seed)
    val_set = MyDataset(root_dir=data_dir, mode="val", transform=data_transforms["val"],
                        split_ratio=args.split_ratio, rng_seed=args.seed)

    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # 创建模型
    model = create_model(args.model_name, num_classes=args.num_classes)
    model.to(device)
    pg = [p for p in model.parameters() if p.requires_grad]

    # 定义优化器
    optimizer = optim.SGD(pg, lr=lr, momentum=args.momentum, weight_decay=weight_decay)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    train(model, train_loader, val_loader, optimizer, loss_fn, scheduler)
    tb_writer.close()
