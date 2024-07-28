import os
import json
import argparse

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import GoogLeNet

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", default="../data/test1/2.jpg",
                    help="path to image")
parser.add_argument("--model_path", default="experiments/base_model/best.pth",
                    help="path to model weight")
class_dict = {"0": "cat", "1": "dog"}
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    img_path = args.img_path
    assert os.path.exists(img_path), "file: '{}' dose not exist!".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)

    # img ---> tensor
    # [N, C, H, W]
    img_tensor = transform(img)
    # [C, H, W]
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    # 加载模型
    model = GoogLeNet(num_classes=2, aux_logits=False).to(device)

    # 加载权重
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.eval()

    with torch.no_grad():
        output = torch.squeeze(model(img_tensor.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {} prob: {:.3f}".format(class_dict[str(predict_cla)], predict[predict_cla].numpy())

    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10} prob: {:.3f}".format(class_dict[str(i)], predict[i].numpy()))

    plt.show()

if __name__ == '__main__':
    main()



