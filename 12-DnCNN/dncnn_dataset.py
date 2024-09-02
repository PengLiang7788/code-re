import torch
from torch.utils.data import Dataset
import utils.utils_image as util


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
        img_H = util.imread_uint(H_path, self.in_channels)
