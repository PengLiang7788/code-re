import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from net import MyAlexNet
from dataset import MyDataset
from torch.utils.tensorboard import SummaryWriter
import time

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集
data_path = u"D:\datasets\Mydata"
# 定义数据转换
transform = transforms.Compose([
    # 中心裁剪
    transforms.CenterCrop(227),
    transforms.ToTensor()
])
test_set = MyDataset(root=data_path, train=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=128)

test_size = len(test_set)

# 定义模型
model = MyAlexNet(num_classes=158)
model.to(device=device)

# 定义损失函数
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.to(device=device)

# 定义优化器
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 定义模型训练参数
epoch = 90
total_train_step = 0
total_test_step = 0

# 创建tensorboard
writer = SummaryWriter('./logs')
# 模型测试
model.eval()
total_test_loss = 0
total_accuracy = 0
with torch.no_grad():
    for data in test_loader:
        total_test_step+=1
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.squeeze(1)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        total_test_loss += loss
        predicted = outputs.max(1)[1]
        accuracy = (predicted == targets).sum()
        total_accuracy += accuracy
print("整体测试集上的Loss: {}".format(total_test_loss))
print("整体测试集上的正确率: {}".format(total_accuracy / test_size))
writer.add_scalar("test_loss", total_test_loss, total_test_step)
writer.add_scalar("test_accuracy", total_accuracy / test_size, total_test_step)

writer.close()
