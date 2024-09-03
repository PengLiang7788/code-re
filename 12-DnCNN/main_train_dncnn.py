import os
import time

import torch
import argparse
import random
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dncnn_dataset import DatasetDnCNN
from torch.utils.data import DataLoader
from model.net import DnCNN
import torch.nn as nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils import utils_image as util
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def parse_opt():
    parser = argparse.ArgumentParser(description="Pytorch DnCNN Training")
    parser.add_argument('--epochs', type=int, default=500, help="number of total epochs to train")
    parser.add_argument('--batch_size', type=int, default=64, help="mini-batch size (default: 128)")
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-4, help="initial learning rate")
    parser.add_argument('--print_freq', '-p', default=1, type=int, help="print frequency (default: 1)")
    parser.add_argument('--model_name', default='dncnn25', type=str, help='name of experiment')
    parser.add_argument("--seed", type=int, default=23, help="random seed")
    parser.add_argument('--workers', type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--train_set", type=str, default="./datasets/Train400", help="path to dataset")
    parser.add_argument("--test_set", type=str, default="./datasets/Test/Set12", help="path to dataset")
    parser.add_argument('--sigma', type=int, default=25, help="noise level")
    parser.add_argument('--sigma_test', type=int, default=25, help="noise level")
    parser.add_argument('--patch_size', type=int, default=40, help="patch size 40 | 64 | 96 | 128 | 192")

    return parser.parse_args()


def seed_torch(seed=123):
    # 设置随机数生成器的种子，确保实验可重复性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model: nn.Module, optimizer: torch.optim, train_loader: DataLoader, loss_fn: nn.L1Loss):
    model.train(True)
    loss_recoder = 0.
    for data in tqdm(train_loader, desc="train"):
        inputs, labels = data['L'].to(device), data['H'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss_recoder += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_recoder


def evaluation(model: nn.Module, test_loader: DataLoader, loss_fn: nn.L1Loss, current_step: int):
    model.eval()
    avg_psnr = 0.0
    loss_recoder = 0.0
    idx = 0
    with torch.no_grad():
        for test_data in tqdm(test_loader, desc="Evaluation"):
            idx += 1
            inputs, labels = test_data['L'].to(device), test_data['H'].to(device)
            image_name_ext = os.path.basename(test_data['L_path'][0])
            img_name, ext = os.path.splitext(image_name_ext)

            img_dir = os.path.join(exp_path, "test", img_name)
            os.makedirs(img_dir, exist_ok=True)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss_recoder += loss.item()

            E_img = outputs.detach()[0].float().cpu()
            H_img = labels.detach()[0].float().cpu()

            E_img = util.tensor2uint(E_img)
            H_img = util.tensor2uint(H_img)

            # save estimated image E
            save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
            util.imsave(E_img, save_img_path)

            # 计算psnr
            current_psnr = compare_psnr(H_img, E_img)
            avg_psnr += current_psnr
    avg_psnr = avg_psnr / idx

    return loss_recoder, avg_psnr


def train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
          optimizer: torch.optim, loss_fn: nn.L1Loss, scheduler: lr_scheduler):
    # 记录训练开始时间
    start_time = time.time()
    # 记录最大psnr, 初始为0
    best_psnr = 0.0
    # 打开一个用于写入训练过程的信息文件
    f = open(os.path.join(exp_path, "lr_{}.txt".format(lr)), "w")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, loss_fn)
        test_loss, psnr = evaluation(model, test_loader, loss_fn, epoch + 1)

        if psnr > best_psnr:
            best_psnr = psnr
            state_dict = dict(epoch=epoch + 1, model=model.state_dict())
            pth_path = os.path.join(exp_path, "ckpt", "best.pth")
            os.makedirs(os.path.dirname(pth_path), exist_ok=True)
            torch.save(state_dict, pth_path)

        # 更新学习率
        scheduler.step()
        # 定义要记录的训练信息标签
        tags = ["train_losses", "test_lossed", "test_psnr"]
        tb_writer.add_scalar(tags[0], train_loss, epoch + 1)
        tb_writer.add_scalar(tags[1], test_loss, epoch + 1)
        tb_writer.add_scalar(tags[2], psnr, epoch + 1)

        # 打印训练过程信息，以及将信息写入文件
        if (epoch + 1) % args.print_freq == 0:  # print_freq指定为1 则每轮都打印
            msg = "epoch:{} model:{} train loss:{:.2f} test loss{:.2f} psnr:{:.2f}\n".format(
                epoch + 1,
                args.model_name,
                train_loss,
                test_loss,
                psnr,
            )
            print(msg)
            f.write(msg)
            f.flush()
    # 输出训练结束后的最佳准确度和总训练时间
    msg_best = "model:{} best psnr:{:.2f}\n".format(args.model_name, best_psnr)
    time_elapsed = "training time: {}".format(time.time() - start_time)
    print(msg_best)
    f.write(msg_best)
    f.write(time_elapsed)
    f.close()


if __name__ == '__main__':
    args = parse_opt()
    # 设置随机数种子, 确保实验的可重复性
    seed_torch(args.seed)
    # 获取数据集存放路径
    train_root = args.train_set
    test_root = args.test_set
    # 设置训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置实验结果保存路径
    # exp/modelName_datetime_lr
    model_name = args.model_name
    lr = args.lr
    now = datetime.now().strftime(r"%mM_%dD_%HH")
    exp_path = "exp/{}_{}_lr{}".format(model_name, now, lr)
    os.makedirs(exp_path, exist_ok=True)

    # 创建tensorboard
    tb_path = os.path.join(exp_path, "runs")
    os.makedirs(tb_path, exist_ok=True)
    tb_writer = SummaryWriter(tb_path)

    # 创建数据加载器
    train_set = DatasetDnCNN(train_root, mode="train", in_channels=1,
                             patch_size=args.patch_size, sigma=args.sigma,
                             sigma_test=args.sigma_test)
    test_set = DatasetDnCNN(test_root, mode="test", in_channels=1,
                            patch_size=args.patch_size, sigma=args.sigma,
                            sigma_test=args.sigma_test)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             num_workers=1, pin_memory=True)

    # 构建模型
    model = DnCNN()
    model.to(device)

    # 定义损失函数
    loss_fn = nn.L1Loss().to(device)
    # 定义优化器
    optim_params = []
    for k, v in model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
    optimizer = Adam(optim_params, lr=lr)
    # 定义scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, [200000, 400000, 600000, 800000, 1000000, 2000000], 0.5)
    # 模型训练
    train(model, train_loader, test_loader, optimizer, loss_fn, scheduler)
    tb_writer.close()
