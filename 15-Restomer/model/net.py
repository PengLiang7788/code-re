import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        """
        Multi-DConv head transposed attention
        Args:
            dim: 输入通道数
            num_heads: 注意力头的数量
            bias: 是否使用偏差
        """
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 可学习系数, 指的是注意力机制中的sqrt(d)

        # 1x1卷积升维
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 3x3分组卷积
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 最后的1x1卷积
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):  # x: (b, dim, h, w)
        b, c, h, w = x.shape  # 输入的结构
        # 计算qkv矩阵
        # qkv: (b, dim, h, w) -> (b, dim * 3, h, w)
        # qkv_dwconv: (b, dim * 3, h, w) -> (b, dim * 3, h, w)
        qkv = self.qkv_dwconv(self.qkv(x))
        # 将qkv切分成q, k, v
        # chunk: (b, dim * 3, h, w) -> (b, dim, h, w)
        q, k, v = qkv.chunk(3, dim=1)
        # 调整q, k, v的形状
        # (b, dim, h, w) -> (b, num_head, c, h*w)
        q = q.reshape(b, self.num_heads, -1, h * w).contiguous()
        k = k.reshape(b, self.num_heads, -1, h * w).contiguous()
        v = v.reshape(b, self.num_heads, -1, h * w).contiguous()

        # 在最后一层进行归一化
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # q @ k
        # transpose(-2, -1): (b, num_head, c, h*w) -> (b, num_head, h*w, c)
        # q @ k.transpose(-2, -1): (b, num_head, c, h*w) @ (b, num_head, h*w, c) = (b, num_head, c, c)
        score = (q @ k.transpose(-2, -1)) * self.temperature
        # softmax: (b, num_head, c, c) -> (b, num_head, c, c)
        score = score.softmax(dim=-1)
        # score 和 v 做矩阵乘法
        # score @ v: (b, num_head, c, c) * (b, num_head, c, h*w) -> (b, num_head, c, h*w)
        out = score @ v
        # reshape: (b, num_head, c, h*w) -> (b, num_head * c, h, w)
        out = out.reshape(b, c, h, w)
        # 1x1卷积 dim = c * num_head
        out = self.project_out(out)
        return out


