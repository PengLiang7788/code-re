import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import os

random.seed(123)


class MyDataset(Dataset):
    def __init__(self, root, train=True, transform=None, split_ratio=0.9, rng_seed=620):
        super(MyDataset, self).__init__()
        self.data = [] # 保存图片路径和标签
        self.data_dir = root # 数据存放路径
        self.rng_seed = rng_seed # 定义随机数种子
        self.transform = transform # 定义数据转换
        self.mode = "train" if train else "val" # 判断是训练还是验证集
        self.data = self._get_data() # _get_data()返回(图片路径,图片标签)构成的列表
        self.split_ratio = split_ratio # 定义训练集和验证集比例
    
    def __getitem__(self, idx):
        path_img, label = self.data[idx]
        img = Image.open(path_img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        if len(self.data) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.data)
    
    def _get_data(self):
        '''
        返回(图片路径,图片标签)构成的列表
        '''
        img_names = os.listdir(self.data_dir)
        img_names = list(filter(lambda x:x.endswith('.jpg'), img_names))

        random.seed(self.rng_seed)
        random.shuffle(img_names)

        # 获取图片标签 0-猫 1-狗
        img_labels = [0 if n.startswith('cat') else 1 for n in img_names]
        # 分割数据集
        split_idx = [int(len(img_labels) * self.split_ratio)]
        if self.mode == 'train':
            img_set = img_names[:split_idx]
            label_set = img_labels[:split_idx]
        elif self.mode == 'val':
            img_set = img_names[split_idx:]
            label_set = img_labels[split_idx:]
        else:
            raise Exception("self.mode 无法识别, 仅支持(train, val)")
        
        path_img = [os.path.join(self.data_dir, n) for n in img_set]
        data = [(n, l) for n, l in zip(path_img, label_set)]
        return data
