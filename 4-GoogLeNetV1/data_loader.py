from torch.utils.data import Dataset, DataLoader
import torch
import random
from PIL import Image
import os
from torchvision import transforms

random.seed(123)
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
normalizes = transforms.Normalize(norm_mean, norm_std)

data_transform = {
    "train": transforms.Compose([
        # 将短边缩放到256
        transforms.Resize((256)),
        # 从中心裁剪图像，大小为 256 x 256
        transforms.CenterCrop(256),
        # 随机裁剪大小为224 x 224 的图像块
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)]),
    "val": transforms.Compose([
        transforms.Resize((256)),
        transforms.CenterCrop(256),
        transforms.TenCrop(224, vertical_flip=False),
        transforms.Lambda(lambda crops: torch.stack([normalizes(transforms.ToTensor()(crop)) for crop in crops]))])
}

class MyDataset(Dataset):
    def __init__(self, root, mode="train", transform=None, split_ratio=0.9, rng_seed=620):
        super(MyDataset, self).__init__()
        self.data = []                          # 存放 (img_path, label)构成的列表
        self.data_dir = root                    # 数据集存放路径
        self.mode = mode                        # 判断是训练集还是验证集
        self.transform = transform              # 定义数据转换
        self.split_ratio = split_ratio          # 定义训练集和验证集比例
        self.rng_seed = rng_seed                # 定义随机数种子
        self.data = self._get_data()            # _get_data()返回(img_path, label)构成的列表
    
    def __getitem__(self, idx):
        path_img, label = self.data[idx]
        img = Image.open(path_img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

    def __len__(self):
        if len(self.data) == 0:
            raise Exception("\n data_dir: {} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.data)

    def _get_data(self):
        """
        返回 (img_path, label)构成的列表
        """
        img_names = os.listdir(self.data_dir)
        img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

        random.seed(self.rng_seed)
        random.shuffle(img_names)

        # 获取图片标签，0-猫 1-狗
        img_labels = [0 if n.startswith('cat') else 1 for n in img_names]
        # 分割数据集
        split_idx = int(len(img_labels) * self.split_ratio)
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

def fetch_dataloader(types, data_dir, params):
    """
    创建DataLoader对象
    Args: 
        types: (list) has one or more of 'train', 'val', 'test' depending on wich data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
        data_len: (dict) contains the Dataset length
    """
    dataloaders = {}
    data_len = {}

    for split in ['train', 'val']:
        if split in types:
            dataset = MyDataset(root=data_dir, mode=split, transform=data_transform[split])
            dl = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)
            dataloaders[split] = dl
            data_len[split] = len(dataset)
    
    return dataloaders, data_len
        


if __name__ == "__main__":
    data_dir = '../data/train'
    train_dataset = MyDataset(root=data_dir, train=True)

