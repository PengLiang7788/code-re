import torch
from torch.utils.data import Dataset
import utils.utils_image as util
import random
import numpy as np


class DatasetDnCNN(Dataset):
    def __init__(self, data_root, mode, in_channels=3, patch_size=64, sigma=25, sigma_test=25):
        """
        Args:
            data_root: path of dataset
            mode: train or test
            in_channels: channels of input image
            sigma: 15, 25, 50 for DnCNN
            sigma_test: 15 25 50 for DnCNN
        """
        super(DatasetDnCNN, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.sigma = sigma
        self.sigma_test = sigma_test
        self.mode = mode

        # 获取图像路径
        self.paths_H = util.get_image_paths(data_root)

    def __getitem__(self, index):
        H_path = self.paths_H[index]
        # 读取原始图像
        img_H = util.imread_uint(H_path, self.in_channels)

        L_path = H_path

        if self.mode == 'train':
            H, W, _ = img_H.shape

            # 从图像中随机裁剪出图像块
            rand_h = random.randint(0, max(0, H - self.patch_size))
            rand_w = random.randint(0, max(0, W - self.patch_size))
            # 裁剪出原始图像块
            patch_H = img_H[rand_h: rand_h + self.patch_size, rand_w:rand_w + self.patch_size, :]

            # 数据增强 flip, rotate
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            # 将HWC格式的图像转换成CHW格式的tensor, 且进行归一化
            img_H = util.uint2tensor3(patch_H)
            # img_L噪声图, img_H干净图
            img_L = img_H.clone()

            # 添加噪声
            noise = torch.randn(img_L.size()).mul_(self.sigma / 255.0)
            img_L.add_(noise)
        else:
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)

            # 添加噪声
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma_test / 255.0, img_L.shape)

            # numpy -> tensor
            img_L = util.single2tensor3(img_L)
            img_H = util.single2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)
