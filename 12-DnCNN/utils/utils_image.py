import os
import cv2
import numpy as np

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
