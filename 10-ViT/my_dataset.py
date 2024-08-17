import random

import torch
from torch.utils.data import Dataset
import os
from typing import List
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, root_dir: str, mode="train", transform=None, split_ratio: int = 0.1,
                 rng_seed: int = 123):
        self.data = []  # 存储(image_path, label)构成的列表
        self.root_dir = root_dir  # 数据集根目录
        self.mode = mode  # train or val
        self.split_ratio = split_ratio  # 验证集所占比例
        self.transform = transform  # 定义数据转换
        self.rng_seed = rng_seed  # 随机数种子
        self.data = self._get_data()  # 返回(image_path, label)构成的标签

    def __len__(self):
        if len(self.data) == 0:
            raise Exception(
                "\n data_dir: {} is a empty dir! Please checkout your path to images!".format(self.root_dir))
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path)
        if image.mode != "RGB":
            raise ValueError("image: {} isn't RGB mode!".format(self.data[idx][0]))
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def _get_data(self):
        # 获取数据集所有类别
        flower_classes = [cla for cla in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, cla))]
        # 排序, 确保各平台顺序一致
        flower_classes.sort()
        # 生成类别名称对应的数字索引
        class_indices = dict((k, v) for v, k in enumerate(flower_classes))

        images_path = []  # 存储训练集的所有图片路径
        images_label = []  # 存储训练集的所有图片标签
        supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
        # 遍历每个文件夹
        for cls in flower_classes:
            cla_path = os.path.join(self.root_dir, cls)
            # 遍历所有支持的文件
            images = [os.path.join(cla_path, i) for i in os.listdir(cla_path)
                      if os.path.splitext(i)[-1] in supported]
            # 排序
            images.sort()
            # 获取类别对应的索引
            image_class = class_indices[cls]
            # 按比例随机采样验证样本
            val_path = random.sample(images, k=int(len(images) * self.split_ratio))
            if self.mode == "train":
                for img_path in images:
                    if img_path not in val_path:
                        images_path.append(img_path)
                        images_label.append(image_class)
            elif self.mode == "val":
                for img_path in images:
                    if img_path in val_path:
                        images_path.append(img_path)
                        images_label.append(image_class)
            else:
                raise Exception("self.mode无法识别, 仅支持(train/val)")
            data = [(n, l) for n, l in zip(images_path, images_label)]
            return data


if __name__ == "__main__":
    root = r"D:\datasets\flower_photos"
    dataset = MyDataset(root_dir=root)
