import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

# 固定数据增强的情况下, 块大小为40, 步长为10, BSD400图像块数量为238336, 与128*1600接近
# 固定数据增强的情况下, 块大小为50, 步长为7, BSD400图像块数量为386688, 与128*3000接近
# 固定数据增强的情况下, 块大小为50, 步长为4, BSD400图像块数量为1146752, 与128*8000接近
patch_size, stride = 40, 10  # 图像块大小, 步长
aug_times = 1  # 每个图像块增强次数
scales = [1, 0.9, 0.8, 0.7]  # 数据增强缩放
batch_size = 128  # mini-batch大小


# 封装带噪声和不带噪声的图像块
class DenoisingDataset(Dataset):
    """
    Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches. (n, c, h, w)四维张量, 在训练前制作好, n是总图像块数量
        sigma: noise level, e.g., 25
    """

    def __init__(self, xs, sigma):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.sigma = sigma

    def __getitem__(self, index):
        batch_x = self.xs[index]  # 每个图像块
        # 噪声生成: 生成与batch_x相同形状满足标准正太分布的张量, 然后按元素乘[0, 255]像素范围内的噪声标准差
        noise = torch.randn(batch_x.size()).mul_(self.sigma / 255.0)
        batch_y = batch_x + noise
        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)


# 展示图像块
def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


# 数据增强选项
def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


# 生成图像块
def gen_patches(file_name):
    # 灰度模式读取图片
    img = cv2.imread(file_name, 0)
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h * s), int(w * s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # 提取图像块, 缩放后按块大小和步长裁剪图像块, 并应用随机数据增强
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = img_scaled[i:i + patch_size, j:j + patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(img, mode=np.random.randint(0, 8))
                    patches.append(x_aug)

    return patches


# 得到训练集中所有的图像块
def data_generator(data_dir='data/Train400', verbose=False):
    # 从数据集中生成干净图像块
    # 获取所有的png文件
    file_list = glob.glob(data_dir + '/*.png')
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:
            data.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')  # 转成numpy, (n, h, w)
    data = np.expand_dims(data, axis=3)  # (n, h, w, 1), 因为网络输入输出通道都是1, 直接添加一维通道数, 满足DataLoader的tensor格式
    discard_n = len(data) - len(data) // batch_size * batch_size  # 多余样本数
    data = np.delete(data, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data  # (n, h, w, 1)


if __name__ == '__main__':
    data = data_generator(data_dir='data/Train400')
    print(len(data))
    print(data.shape)
