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
trans_train = transforms.Compose([
    transforms.Resize((256, 256)),
    # 随机裁剪
    transforms.RandomResizedCrop(227),
    # 随机水平翻转
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
trans_test = transforms.Compose([
    transforms.Resize((256, 256)),
    # 中心裁剪
    transforms.CenterCrop(227),
    transforms.ToTensor()
])
train_set = MyDataset(root=data_path, train=True, transform=trans_train)
test_set = MyDataset(root=data_path, train=False, transform=trans_test)
train_loader = DataLoader(train_set, batch_size=128)
test_loader = DataLoader(test_set, batch_size=128)

train_size = len(train_set)
test_size = len(test_set)

print("训练集长度:{}".format(train_size))
print("测试集长度:{}".format(test_size))

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
for i in range(epoch):
    print("=================第 {} 轮 训练开始=================".format(i + 1))
    model.train(True)
    start_time = time.time()
    for data in train_loader:
        start_time = time.time()
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.squeeze(1)
        targets = targets.to(device)

        outputs = model(imgs)

        # 计算训练误差
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录训练次数
        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(
                "训练时间为: {}, 训练次数为: {}, Loss: {}".format(end_time - start_time, total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss)

    # 模型测试
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            total_test_step += 1
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.squeeze(1)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.max(1)[1] == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_size, total_test_step)

    # 保存模型
    torch.save(model.state_dict(), "./model/model_{}.pth".format(i))
    print("模型{}保存成功".format(i))

writer.close()
