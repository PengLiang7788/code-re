import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers


# (b, c, h, w) -> (b, h*w, c)
def to_3d(x):
    b, c = x.shape[:2]
    x = x.reshape(b, c, -1).transpose(1, 2)
    return x


# (b, h*w, c) -> (b, c, h, w)
def to_4d(x, h, w):
    b = x.shape[0]
    x = x.reshape(b, -1, h, w)
    return x


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # (b,h*w,c)
        sigma = x.var(-1, keepdim=True, unbiased=False)  # 计算矩阵x沿着最后一个维度的方差
        '''
        var: 计算方差的函数
        -1: 表示最后一个维度
        keepdim=True 表示保留维度
        unbiased = False 表示使用有偏方差的计算方式
        '''
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)  # 计算均值
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias  # 添加偏置


class LayerNorm(nn.Module):  # 层归一化
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):  # (b,c,h,w)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
        # to_3d后：(b,h*w,c)
        # body后：(b,h*w,c)
        # to_4d后：(b,c,h,w)


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
    def __init__(self, dim, num_heads, expand_factor=4, bias=False, LayerNorm_type='WithBias'):
        """
        标准Transformer架构
        Args:
            dim: 输入维度
            num_heads: 多头的数量
            expand_factor: 扩张因子
            bias: 是否使用偏置
            LayerNorm_type: 层归一化类型
        """
        super(TransformerBlock, self).__init__()

        # 层正则化
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
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
    def __init__(self, in_channels=3, dim=48, num_heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66, bias=False,
                 num_blocks=[4, 6, 6, 8], num_refinement_blocks=4):
        """
        Restormer Network
        Args:
            in_channels: 输入图像通道数
            dim: 将图像转换成块嵌入维度
            num_heads: 多头的数量
            ffn_expansion_factor: 前馈网络隐藏层扩张因子
            bias: 是否使用偏差
            num_blocks: 每个transformer block个数
            num_refinement_blocks: 最后一个transformer block个数
        """
        super(Restormer, self).__init__()
        # 3x3 patch-embed
        self.patch_embed = nn.Conv2d(in_channels=in_channels, out_channels=dim,
                                     kernel_size=3, stride=1, padding=1, bias=bias)

        # L1 transformer encoder
        self.encoder_l1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=num_heads[0], expand_factor=ffn_expansion_factor, bias=bias) for _ in
              range(num_blocks[0])])
        self.down1 = Downsample(dim)

        # L2 transformer encoder
        self.encoder_l2 = nn.Sequential(
            *[TransformerBlock(dim=dim * 2, num_heads=num_heads[1], expand_factor=ffn_expansion_factor, bias=bias) for _
              in range(num_blocks[1])])
        self.down2 = Downsample(dim * 2)

        # L3 transformer encoder
        self.encoder_l3 = nn.Sequential(
            *[TransformerBlock(dim=dim * 4, num_heads=num_heads[2], expand_factor=ffn_expansion_factor, bias=bias) for _
              in range(num_blocks[2])])
        self.down3 = Downsample(dim * 4)

        # L4 transformer block
        self.latent = nn.Sequential(
            *[TransformerBlock(dim=dim * 8, num_heads=num_heads[3], expand_factor=ffn_expansion_factor, bias=bias) for _
              in range(num_blocks[3])])

        # 上采样
        self.upsample1 = Upsample(dim=dim * 8)
        # reduce channels
        self.reduce1 = nn.Conv2d(in_channels=dim * 8, out_channels=dim * 4, kernel_size=1, bias=bias)
        self.decoder_l3 = nn.Sequential(
            *[TransformerBlock(dim=dim * 4, num_heads=num_heads[2], expand_factor=ffn_expansion_factor, bias=bias) for _
              in range(num_blocks[2])])

        # 上采样
        self.upsample2 = Upsample(dim=dim * 4)
        # reduce channels
        self.reduce2 = nn.Conv2d(in_channels=dim * 4, out_channels=dim * 2, kernel_size=1, bias=bias)
        self.decoder_l2 = nn.Sequential(
            *[TransformerBlock(dim=dim * 2, num_heads=num_heads[1], expand_factor=ffn_expansion_factor, bias=bias) for _
              in range(num_blocks[1])])

        # 上采样
        self.upsample3 = Upsample(dim=dim * 2)
        self.decoder_l1 = nn.Sequential(
            *[TransformerBlock(dim=dim * 2, num_heads=num_heads[0], expand_factor=ffn_expansion_factor, bias=bias) for _
              in range(num_blocks[0])])

        # refinement
        self.refinement = nn.Sequential(
            *[TransformerBlock(dim=dim * 2, num_heads=num_heads[0], expand_factor=ffn_expansion_factor, bias=bias) for _
              in range(num_refinement_blocks)])

        self.output = nn.Conv2d(in_channels=dim * 2, out_channels=3, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):  # (b, 3, h, w)
        # (b, c, h, w)
        in_enc_level1 = self.patch_embed(x)
        # encoder level1 (b, c, h, w)
        out_enc_level1 = self.encoder_l1(in_enc_level1)
        # downsample (b, 2c, h/2, w/2)
        in_enc_level2 = self.down1(out_enc_level1)
        # encoder level2 (b, 2c, h/2, w/2)
        out_enc_level2 = self.encoder_l2(in_enc_level2)
        # downsample (b, 4c, h/4, w/4)
        in_enc_level3 = self.down2(out_enc_level2)
        # encoder level3 (b, 4c, h/4, w/4)
        out_enc_level3 = self.encoder_l3(in_enc_level3)
        # downsample (b, 8c, h/8, w/8)
        in_enc_level4 = self.down3(out_enc_level3)
        # level4 (b, c*8, h/8, w/8)
        latent = self.latent(in_enc_level4)

        # upsample (b, c*4, h/4, w/4)
        in_dec_level3 = self.upsample1(latent)
        # concat (b, c*8, h/4, w/4)
        in_dec_level3 = torch.cat([out_enc_level3, in_dec_level3], dim=1)
        # reduce dimension (b, c*4, h/4, w/4)
        in_dec_level3 = self.reduce1(in_dec_level3)
        # decoder level3 (b, c*4, h/4, w/4)
        out_dec_level3 = self.decoder_l3(in_dec_level3)

        # upsample (b, c*2, h/2, w/2)
        in_dec_level2 = self.upsample2(out_dec_level3)
        # concat (b, c*4, h/2, w/2)
        in_dec_level2 = torch.cat([out_enc_level2, in_dec_level2], dim=1)
        # reduce dimension (b, c*2, h/2, w/2)
        in_dec_level2 = self.reduce2(in_dec_level2)
        # decoder level2 (b, c*2, h/2, w/2)
        out_dec_level2 = self.decoder_l2(in_dec_level2)

        # upsample (b, c, h, w)
        in_dec_level1 = self.upsample3(out_dec_level2)
        # concat (b, 2c, h, w)
        in_dec_level1 = torch.cat([out_enc_level1, in_dec_level1], dim=1)
        # decoder level1 (b, 2c, h, w)
        out_dec_level1 = self.decoder_l1(in_dec_level1)

        # refinement (b, 2c, h, w)
        refinement = self.refinement(out_dec_level1)

        # output (b, 3, h, w)
        output = self.output(refinement)

        return output + x


if __name__ == "__main__":
    model = Restormer()
    print(model)

    x = torch.randn((1, 3, 64, 64))
    x = model(x)
    print(x.shape)
