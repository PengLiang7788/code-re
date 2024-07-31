import torch
from model import model_dict
import argparse
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='resnet18',
                    help="model name default resnet18")
parser.add_argument('--image_path', type=str, default='../data/test1/2.jpg',
                    help="predict image path")
parser.add_argument('--classes_num', type=int, default=2)
parser.add_argument('--model_path', type=str, default='runs/report/resnet18/seed33/ckpt/best.pth')

class_dict = {"0": "cat", "1": "dog"}

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    img_path = args.image_path
    assert os.path.exists(img_path), "file: '{}' dose not exist!".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)

    img_tensor = transform(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    # 加载模型
    model = model_dict[args.model_name](num_classes=args.classes_num)
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

