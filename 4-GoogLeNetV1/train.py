import argparse
import os
import sys
import logging

import torch
from torch.autograd import Variable
import torch.utils
from model import GoogLeNet
from tqdm import tqdm
import utils
import numpy as np
import data_loader
from torch import optim

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/train', 
                    help="Directory countaining the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")
parser.add_argument('--log_dir', default='logs',
                    help="Direcotry containing tensorboard file")


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, params, model_dir, num_val):
    """
    Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.util.data.DataLoader object fetches training data
        val_dataloader: (DataLoader) a torch.util.data.DataLoader object fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        params: (Params) hyperparamaters
        model_dir: (string) directory containing config, weights and log
    """
    model.to(device)
    loss_fn.to(device)

    best_acc = 0.0
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    save_path = os.path.join(model_dir, 'best.pth')
    train_steps = len(train_dataloader)

    for epoch in range(params.num_epochs):
        # train
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_dataloader, file=sys.stdout)
        for step, data in enumerate(train_dataloader):
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            logits, aux_logits2, aux_logits1 = model(imgs)
            loss0 = loss_fn(logits, labels)
            loss1 = loss_fn(aux_logits1, labels)
            loss2 = loss_fn(aux_logits2, labels)
            loss = loss0 + loss1 * 0.3 + loss2 + 0.3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 输出统计信息
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1, params.num_epochs, loss)
        
        # validate
        model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                bs, ncrops, c, h, w = val_images.size()

                outputs = model(val_images.view(-1, c, h, w))
                outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
                predict = torch.max(outputs_avg, dim=1)[1]
                acc += torch.eq(predict, val_labels).sum().item()
        
        val_accurate = acc / num_val
        print('[epoch %d] train_loss: %.3f val_accuracy: %.3f' % (epoch+1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)
    
    print('Finished Training')




if __name__ == '__main__':

    # 从json file中提取参数
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # 定义训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.cuda = torch.cuda.is_available()

    # 设置随机种子
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # 设置 logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # 加载训练数据集
    logging.info("Loading the datasets...")
    dataloaders, data_len = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    num_train = data_len['train']
    num_val = data_len['val']

    logging.info("- done.")

    # 定义模型和优化器
    model = GoogLeNet(num_classes=2, aux_logits=True, init_weights=True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # 定义损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device)
    
    # 训练模型
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, params, args.model_dir, num_val)





