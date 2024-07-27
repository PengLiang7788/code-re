import argparse
import os

import torch
from model import GoogLeNet

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

