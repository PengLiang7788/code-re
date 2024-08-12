import os.path
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, root: str, mode="train", transform=None, split_ratio=0.9, rng_seed=123):
        self.data = []  # 存储(image_path, label)构成的列表
        self.data_dir = root  # 数据集根目录
        self.mode = mode  # 训练集还是验证集
        self.transform = transform  # 定义数据转换方式
        self.split_ratio = split_ratio  # 数据集于验证集分割比例
        self.rng_seed = rng_seed  # 随机数种子
        self.data = self._get_data()  # 返回(image_path, label)构成的标签

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        if len(self.data) == 0:
            raise Exception(
                "\n data_dir: {} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.data)

    def _get_data(self):
        image_names = os.listdir(self.data_dir)
        image_names = list(filter(lambda x: x.endswith('.jpg'), image_names))

        # 打乱顺序
        random.seed(self.rng_seed)
        random.shuffle(image_names)
        # 0-猫 1-狗
        labels = [0 if n.startswith('cat') else 1 for n in image_names]

        # 分割数据集
        split_index = int(self.split_ratio * len(labels))

        if self.mode == "train":
            image_set = image_names[:split_index]
            label_set = labels[:split_index]
        elif self.mode == "val":
            image_set = image_names[split_index:]
            label_set = labels[split_index:]
        else:
            raise Exception("self.mode无法识别, 仅支持(train/val)")

        image_set = [os.path.join(self.data_dir, n) for n in image_names]
        data = [(n, l) for n, l in zip(image_set, label_set)]

        return data
