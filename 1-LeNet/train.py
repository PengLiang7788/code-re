import torchvision
import torch
from torch.utils.data import DataLoader
from model import MyModel
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time

# 定义训练的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集准备
train_dataset = torchvision.datasets.MNIST(root='./data',train=True, download=True, 
                                           transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, 
                                          transform=torchvision.transforms.ToTensor())

train_data_size = len(train_dataset)
test_data_size = len(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

print(f"训练数据集长度为: {train_data_size}")
print(f"测试数据集长度为: {test_data_size}")
img, label = train_dataset[0]
print(img.size())

# 定义网络模型
model = MyModel()
model.to(device=device)
# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device=device)
# 定义优化器
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 定义模型训练参数
# 记录训练轮数
epoch = 10
# 记录总的训练次数
total_train_step = 0
# 记录总的测试次数
total_test_step = 0

# 创建 tensorboard
writer = SummaryWriter('./logs')

for i in range(epoch):
    print("================第 {} 轮训练开始================".format(i+1))
    model.train(True)
    start_time = time.time()
    for data in train_loader:
        start_time = time.time()
        imgs, targets = data
        imgs = imgs.to(device)
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
        if total_train_step % 300 == 0:
            end_time = time.time()
            print("训练时间为: {}, 训练次数为: {}, Loss: {}".format(end_time - start_time, total_train_step, loss.item()))
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
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)

    # 保存模型
    torch.save(model.state_dict(), "./model/model_{}.pth".format(i))
    print("模型{}已保存".format(i))

writer.close()



