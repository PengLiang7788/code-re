import numpy as np
import torch


class AverageMeter(object):
    def __init__(self):
        """
        计算并存储当前值和平均值的工具类
        """
        self.reset()

    def reset(self):
        self.val = 0.0  # 当前值
        self.count = 0  # 计数器
        self.avg = 0.0  # 平均值
        self.sum = 0.0  # 总和

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count


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
    pred = pred.t()
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
