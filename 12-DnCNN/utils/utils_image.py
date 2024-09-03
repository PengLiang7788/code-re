import os
import cv2
import numpy as np
import torch

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename: str):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_paths(data_root):
    """
        从给出的数据集跟路径中获取所有图像的路径
    """
    paths = None
    if isinstance(data_root, str):
        paths = sorted(_get_paths_from_images(data_root))
    elif isinstance(data_root, list):
        paths = []
        for i in data_root:
            paths += sorted(_get_paths_from_images(i))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def imread_uint(path, img_channels=3):
    """
    Args:
        path: path of input image
        img_channels: channels of input image
    Returns:
        HxWx3(RGB or GGG), or HxWx1(G)
    """
    if img_channels == 1:
        img = cv2.imread(path, 0)  # 以灰度图方式读取图像
        img = np.expand_dims(img, axis=2)  # 以灰度图读取维度为2: HxW, 添加一个维度变成HxWx1
    elif img_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

    return img


# 数据增强 flip or rotate
def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


# 将HWC转换成三维张量
def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # np.ascontiguousarray(img)将输入图像转换为一个连续的内存块, 对于将Numpy数组转换成PyTorch张量是必要的, 提高转换效率
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)


# convert 2/3/4-dimensional torch tensor to uint
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8(img * 255.0).round()


def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


# convert single (HxWxC) to 3-dimensional torch tensor
def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


def uint2single(img):
    return np.float32(img / 255.)
