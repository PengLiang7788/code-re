import argparse
import os
import logging

import torch
from torch.autograd import Variable
from model import GoogLeNet
from tqdm import tqdm
import utils
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/train', 
                    help="Directory countaining the dataset")
parser.add_argument('--mode_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")
parser.add_argument('--log_dir', default='logs',
                    help="Direcotry containing tensorboard file")

def train(model, optimizer, loss_fn, dataloader, device, metrics, params):
    """
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetchs training data
        device: model training equipment
        metrics: (dict) a dictionary of functions that compute a metrix using the output and labels of each batch
        params: (Params) hyperparmeters
    """
    # 设置模型处于训练状态
    model.train()
    # 对于当前训练循环的总结和对于损失的运行平均对象
    summ = []
    loss_avg = utils.RunningAverage()

    # 使用tqdm进度条
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # 将数据移动到指定设备上
            train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)
            # 转换成 tensor 类型变量
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # 计算模型输出和损失值
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if i % params.save_summary_steps == 0:
                # 从tensor变量中提取数据，移动到cpu，转换为numpy数组
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # 计算在此batch上的所有指标
                summary_batch = {metric:metrics[metric](output_batch, labels_batch) for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)
            
            # 更新平均损失
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    
    # 计算所有指标的均值
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir, restore_file=None):
    pass


if __name__ == '__main__':

    # 从json file中提取参数
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # 定义训练设备
    


