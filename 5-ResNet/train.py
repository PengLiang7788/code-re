import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import model
import numpy as np
import time
import random

# 导入argparse模块，用于解析命令行参数
parser = argparse.ArgumentParser()

parser.add_argument("--model_names", type=str, default="resnet18", help="model name default resnet18")
parser.add_argument("--pre_trained", type=bool, default=False, help="use pre-trained model, default is False")
parser.add_argument("--dataset", type=str, default="../data/train", help="path to dataset")
parser.add_argument("--classes_num", type=int, default=10, help="number of classes")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epoch", type=int, default=20, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
parser.add_argument("--seed", type=int, default=33, help="random seed")
parser.add_argument("--print_freq", type=int, default=1, help="print training message frequency")
parser.add_argument("--exp_postfix", type=str, default="seed33", help="experiment result postfix")
parser.add_argument("--txt_name", type=str, default="lr0.01_wd5e-4", help="experiment result txt name")

args = parser.parse_args()


def seed_torch(seed=74):
    # 设置随机数生成器的种子，确保实验可重复性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_torch(seed=args.seed)

# 从命令行参数中获取实验名称后缀
exp_name = args.exp_postfix

# 创建实验结果文件夹路径
exp_path = "./report/{}/{}/{}".format(args.dataset, args.model_names, exp_name)
os.makedirs(exp_path, exist_ok=True)

# 将图片随即缩放到指定区间中
def random_scale_size(min_size, max_size, current_size):
    # 随机选择短边的大小
    target_short_size = random.randint(min_size, max_size)
    # 计算缩放比例
    if current_size[0] < current_size[1]:
        scale = target_short_size / current_size[0]
    else:
        scale = target_short_size / current_size[1]
    
    # 计算新尺寸
    new_size = (int(current_size[0] * scale), int(current_size[1] * scale))
    return new_size



# 加载数据集
# 定义数据转换
data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop()
    ])
}
train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=data_transform['train'])
