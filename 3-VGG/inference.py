import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from datasets import MyDataset
from model import MyVGG
import torchvision
from torchvision import transforms
from PIL import Image

data_dir = '../data/train'
model_path = './model/model_9.pth'

img_path = os.path.join(data_dir, "cat.34.jpg")

# 加载模型
model = MyVGG()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

# path --> img
img_rgb = Image.open(img_path).convert('RGB')

# img --> tensor
inference_tranform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

img_tensor = inference_tranform(img_rgb)

img_tensor.unsqueeze_(0)


output = model(img_tensor)
_, predicted = torch.max(output, 1)
if predicted.item() == 0:
    print("cat")
else:
    print("dog")



