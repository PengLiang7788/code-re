import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from model import model_dict
import numpy as np
import time
import random
from dataset import MyDataset
import torch.nn as nn
from utils import AverageMeter, accuracy  # 自定义工具模块，用于计算模型的平均值和准确度
from torch.optim.lr_scheduler import CosineAnnealingLR

# 导入argparse模块，用于解析命令行参数
parser = argparse.ArgumentParser()

parser.add_argument("--model_names", type=str, default="resnet18", help="model name default resnet18")
parser.add_argument("--pre_trained", type=bool, default=False, help="use pre-trained model, default is False")
parser.add_argument("--dataset", type=str, default="../data/train", help="path to dataset")
parser.add_argument("--classes_num", type=int, default=2, help="number of classes")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epoch", type=int, default=20, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
parser.add_argument("--seed", type=int, default=33, help="random seed")
parser.add_argument("--print_freq", type=int, default=1, help="print training message frequency")
parser.add_argument("--exp_postfix", type=str, default="seed33", help="experiment result postfix")
parser.add_argument("--txt_name", type=str, default="lr0.01_wd5e-4", help="experiment result txt name")
parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")

args = parser.parse_args()


def seed_torch(seed=123):
    # 设置随机数生成器的种子，确保实验可重复性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_torch(seed=args.seed)

# 从命令行参数中获取实验名称后缀
exp_name = args.exp_postfix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建实验结果文件夹路径
exp_path = "./report/{}/{}/{}".format(args.dataset, args.model_names, exp_name)
os.makedirs(exp_path, exist_ok=True)

# 加载数据集
# 定义数据转换
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

data_transform = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
}

# 创建数据集
train_set = MyDataset(args.dataset, mode="train", transform=data_transform["train"], rng_seed=args.seed)
val_set = MyDataset(args.dataset, mode="val", transform=data_transform["val"], rng_seed=args.seed)

# 创建数据集加载器
# pin_memory 如果可用，将数据加载到 GPU 内存中以提高训练速度
train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

# train
def train_one_epoch(model: nn.Module, optimizer: torch.optim, train_loader: DataLoader, loss_fn: torch.nn.CrossEntropyLoss):
    model.train()
    acc_recorder = AverageMeter()  # 用于记录精度的工具
    loss_recorder = AverageMeter()  # 用于记录损失的工具

    for (inputs, targets) in tqdm(train_loader, desc="train"):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss_recorder.update(loss.item(), n=inputs.size(0))
        acc = accuracy(outputs, targets)[0]  # 计算精度
        acc_recorder.update(acc.item(), n=inputs.size(0))  # 记录精度值
        optimizer.zero_grad()  # 清零之前的梯度
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
    
    losses = loss_recorder.avg  # 计算平均损失
    acces = acc_recorder.avg  # 计算平均精度

    return losses, acces

def evaluation(model: nn.Module, val_loader: DataLoader):
    model.eval()
    acc_recorder = AverageMeter()  # 初始化两个计量器，用于记录准确度和损失
    loss_recorder = AverageMeter()

    with torch.no_grad():
        for img, label in tqdm(val_loader, desc="Evaluating"):
            img, label = img.to(device), label.to(device)
            out = model(img)
            acc = accuracy(out, label)[0]
            loss = F.cross_entropy(out, label) # 计算交叉熵损失
            acc_recorder.update(acc.item(), img.size(0))  # 更新准确率记录器，记录当前批次的准确率  img.size(0)表示批次中的样本数量
            loss_recorder.update(loss.item(), img.size(0))  # 更新损失记录器，记录当前批次的损失
    losses = loss_recorder.avg # 计算所有批次的平均损失
    acces = acc_recorder.avg # 计算所有批次的平均准确率
    return losses, acces # 返回平均损失和准确率

def train(model:nn.Module, train_loader:DataLoader, val_loader:DataLoader, 
          optimizer: torch.optim, scheduler: CosineAnnealingLR, loss_fn: torch.nn.CrossEntropyLoss):
    # 记录训练开始时间
    start_time = time.time()
    # 初始最佳准确率为-1，以便跟踪最佳模型
    best_acc = -1
    # 打开一个用于写入训练过程信息的文件
    f = open(os.path.join(exp_path, "{}.txt".format(args.txt_name)), "w")

    for epoch in range(args.epoch):
        # 在训练集上执行一个周期训练, 并获取训练损失和准确率
        train_losses, train_access = train_one_epoch(model, optimizer, train_loader, loss_fn)
        # 在测试集上评估模型性能，获取测试损失和准确率
        val_losses, val_access = evaluation(model, val_loader)
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
                args.model_names,
                train_losses,
                train_access,
                val_losses,
                val_access,
            )
            print(msg)
            f.write(msg)
            f.flush()
    # 输出训练结束后的最佳准确度和总训练时间
    msg_best = "model:{} best acc:{:.2f}\n".format(args.model_names, best_acc)
    time_elapsed = "traninng time: {}".format(time.time() - start_time)
    print(msg_best)
    f.write(msg_best)
    f.write(time_elapsed)
    f.close()

if __name__ == '__main__':
    tb_path = "runs/{}/{}/{}".format(args.dataset, args.model_names,  # 创建 TensorBoard 日志目录路径
                                     args.exp_postfix)
    tb_writer = SummaryWriter(log_dir=tb_path)
    lr = args.lr
    model = model_dict[args.model_names](num_classes=args.classes_num, pretrained=args.pre_trained)  # 根据命令行参数创建神经网络模型
    model.to(device)

    optimizer = optim.SGD(  # 创建随机梯度下降 (SGD) 优化器
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)  # 创建余弦退火学习率调度器  自动调整lr
    train(model, train_loader, val_loader, optimizer, scheduler, loss_fn)
