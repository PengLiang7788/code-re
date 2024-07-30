import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random


class MyDataset(Dataset):
    def __init__(self, data_dir, mode="train", transform=None, split_ratio=0.9, rng_seed=123):
        """
        Args: 
            data_dir: (str) path to dataset
            mode: (str) train or val
            transform: (torchvision.transforms) image conversion
            split_ratio: (int) trainging sets segmentation ratio
            rng_seed: (int) random seed
        """
        super(MyDataset, self).__init__()
        self.data = []                      # 存放(image_path, label)所构成的列表
        self.data_dir = data_dir            # 数据集存放路径
        self.mode = mode                    # 训练集还是验证集
        self.transform = transform          # 图像转换
        self.split_ratio = split_ratio      # 数据集分割比例
        self.rng_seed = rng_seed            # 随机数种子

        self.data = self._get_data()        # _get_data()返回(img_path, label)构成的列表
    
    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        img = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

    def __len__(self):
        if len(self.data) == 0:
            raise Exception("\n data_dir: {} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.data)


    # 返回(image_path, label)构成的列表
    def _get_data(self):
        image_names = os.listdir(self.data_dir)
        image_names = list(filter(lambda x: x.endswith('.jpg'), image_names))

        random.seed(self.rng_seed)
        random.shuffle(image_names)

        # 获取图像标签 0-cat 1-dog
        image_labels = [0 if x.startswith('cat') else 1 for x in image_names]

        # 分割数据集
        split_index = int(len(image_labels) * self.split_ratio)
        if self.mode == 'train':
            image_set = image_names[:split_index]
            label_set = image_labels[:split_index]
        elif self.mode == 'val':
            image_set = image_names[split_index:]
            label_set = image_labels[split_index:]
        else:
            raise Exception("self.mode 无法识别, 仅支持(train, val)")
        
        path_image = [os.path.join(self.data_dir, n) for n in image_set]
        data = [(n, l) for n, l in zip(path_image, label_set)]

        return data
        

if __name__ == '__main__':
    data_dir = '../data/train'
    dataset = MyDataset(data_dir)
