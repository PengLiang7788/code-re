import os.path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(MyDataset, self).__init__()
        self.data = []
        self.transform = transform
        if train:
            data_txt = os.path.join(root, 'train.txt')
        else:
            data_txt = os.path.join(root, 'test.txt')
        with open(data_txt, 'r') as f:
            for line in f:
                info = line.strip().split(' ')
                if len(info) > 0:
                    self.data.append([os.path.join(root, info[0]), info[1].strip()])

    def __getitem__(self, idx):
        img_file, label = self.data[idx]
        img = Image.open(img_file)
        if self.transform is not None:
            img = self.transform(img)
        label = np.array([label], dtype='int64')

        return img, label


    def __len__(self):
        return len(self.data)


