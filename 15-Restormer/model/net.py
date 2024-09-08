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


class FeedForward(nn.Module):
    def __init__(self, dim, expand_factor=4, bias=False):
        """
        Gated-DConv Feed-Forward Network
        Args:
            dim: 输入维度
            expand_factor: 扩张因子
            bias: 是否使用偏差
        """
        super(FeedForward, self).__init__()
        hidden_dim = int(dim * expand_factor)
        # point-wise convolution 1x1卷积 升维
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        # depth-wise convolution 分组卷积
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3,
                                stride=1, padding=1, bias=bias, groups=hidden_dim * 2)
        # 1x1 convolution 降维
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):  # x: (b, c, h, w)
        # 论文图中在这一步之前已经切成两条并行路,
        # 但都同时进行逐点卷积和深度卷积, 将两步进行完再切分效率更高
        # project_in: (b, c, h, w) -> (b, hidden_dim * 2, h, w)
        x = self.project_in(x)
        # 在通道方向上进行切分
        # dwconv: (b, hidden_dim * 2, h, w) -> (b, hidden_dim * 2, h, w)
        # chunk: (b, hidden_dim * 2, h, w) -> (b, hidden_dim, h, w)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # element-wise multiplication
        x = F.gelu(x1) * x2
        # (b, hidden_dim, h, w) -> (b, c, h, w)
        x = self.project_out(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, expand_factor=4, bias=False):
        """
        标准Transformer架构
        Args:
            dim: 输入维度
            num_heads: 多头的数量
            expand_factor: 扩张因子
            bias: 是否使用偏置
        """
        super(TransformerBlock, self).__init__()

        # 层正则化
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, expand_factor, bias)

    def forward(self, x):
        # LN -> MDTA -> residual
        x = x + self.attn(self.norm1(x))
        # LN -> GDFN -> residual
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    def __init__(self, dim, downscale_factor=2):
        """
        下采样操作, 使用pixel-unshuffle
        Args:
            dim: 输入通道数
            downscale_factor: 下采样倍数
        """
        super(Downsample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(downscale_factor)
        )
    
    def forward(self, x):
        # x: (b, c, h, w)
        # conv2d: (b, c, h, w) -> (b, c//2, h, w)
        # pixelUnshuffle: (b, c//2, h, w) -> (b, c*2, h/2, w/2)
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, dim, upscale_factor=2):
        """
        上采样操作, 使用pixel-shuffle
        Args:
            dim: 输入通道数
            upscale_factor: 上采样倍数
        """
        super(Upsample, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor)
        )
    
    def forward(self, x):
        # x: (b, c, h, w)
        # conv2d: (b, c, h, w) -> (b, c*2, h, w)
        # pixelShuffle: (b, c*2, h, w) -> (b, c//2, h*2, w*2)
        return self.up(x)


class Restormer(nn.Module):
    def __init__(self, in_channels, dim, num_heads, expand_factor=4, bias=False):
        """
        Restormer Network
        Args:
            in_channels: 输入图像通道数
            dim: 将图像转换成块嵌入维度
            num_heads: 多头的数量
            expand_factor: 前馈网络隐藏层扩张因子
            bias: 是否使用偏差
        """
        super(Restormer, self).__init__()
        # 3x3 深度卷积
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, 
                               stride=1, padding=1, bias=bias, groups=dim)
        # L1 transformer block
        self.l1 = TransformerBlock(dim=dim, num_heads=num_heads, expand_factor=expand_factor, bias=bias)
