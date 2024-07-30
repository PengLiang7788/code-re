import os
import torch
import torch.nn.functional as F
import logging
import numpy as np
from torch.nn import init

class AverageMeter(object):
    """
    计算并存储平均值和当前值的工具类
    """
    def __init__(self):
        # 初始化计数器
        self.reset()
    
    def reset(self):
        """
        重置计数器
        """
        self.count = 0   # 样本计数
        self.sum = 0.0   # 样本损失总和
        self.val = 0.0   # 当前损失值
        self.avg = 0.0   # 平均损失值
    
    def update(self, val, n=1):
        """
        更新计数器的值
        Args:
            val: 当前批次的损失值
            n: 当前批次的样本数量, 默认为1
        """
        self.val = val                   # 更新当前损失值
        self.sum += val * n              # 累加总损失之 
        self.count += n                  # 更新样本计数
        self.avg = self.sum / self.count # 计算平均损失值

# topk=(1,)只考虑最高分对应的预测结果精度
def accuracy(output, target, topk=(1,)):
    """
    Compute the precision@k for the specified values of k
    Args:
        output: 模型的输出结果
        target: 真实的目标标签
        topk: 一个元组, 包含要计算精度的top k值, 默认为(1,)
    Returns:
        res: 一个列表, 包含top k值对应的精度百分比
    """
    maxk = max(topk) # 获取要计算的top k值中最大的一个
    batch_size = target.size(0)  # 获取批次大小，即目标标签的数量
    # 从模型输出中获取前k个最高分的类别预测结果
    _, pred = output.topk(maxk, 1, True, True)
    # 转置, 使得每一列对应一个样本预测结果
    pred = pred.T()
    # 将预测结果与目标标签进行比较，得到一个布尔类型的矩阵，表示每个预测是否正确
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []  # 存储计算得到的精度百分比
    for k in topk:  # 遍历要计算的top k值
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True) # 获取前k个预测结果中正确的个数，并将其转换为百分比形式，添加到结果列表中
        res.append(correct_k.mul_(100.0 / batch_size))  # 计算精度百分比，并将结果添加到列表中
    return res # 返回各个top k值对应的精度百分比

def accuracy_pskd(output, target):
    """Computes the precision@k for the specified values of k"""
    total = 0
    correct = 0

    _, predicted = torch.max(output, 1)
    total += target.size(0)
    correct += predicted.eq(target).sum().item()   # 计算分类精度，即正确分类的样本数占总样本数的比例

    return correct / total