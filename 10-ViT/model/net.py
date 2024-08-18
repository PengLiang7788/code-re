import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        """
        将输入的图像转换成patch
        Args:
            image_size: 输入图像尺寸
            patch_size: patch尺寸
            in_c: 输入通道数
            embed_dim: embedding 维度, dimension = patch_size * patch_size * in_c
            norm_layer: 正则化层
        """
        super(PatchEmbed, self).__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        # 图像尺寸
        self.image_size = image_size
        # patch尺寸
        self.patch_size = patch_size
        # 完整图像划分成patch的尺寸,
        # 如image_size = (8, 8), patch_size = (4, 4), 则grid_size = (2, 2)
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        # 可以划分的总的patch数
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 使用二维卷积进行转换
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        assert h == self.image_size[0] and w == self.image_size[1], \
            f"Input image size ({h}*{w}) doesn't match model ({self.image_size[0] * self.image_size[1]}).)"

        x = self.proj(x)
        # [b, c, h, w] ==> [b, c, hw]
        x = torch.flatten(x, 2)
        # [b, c, hw] ==> [b, hw, c]
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


# MLP
class FeedForward(nn.Module):
    def __init__(self, in_planes, hidden_planes=None, out_planes=None, dropout=0.):
        super().__init__()
        out_planes = out_planes or in_planes
        hidden_planes = hidden_planes or in_planes
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, hidden_planes),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_planes, out_planes),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768,  # 输入token维度
                 num_head=8,  # 多头的数量
                 qkv_bias=False,  # 将q,k,v转换成矩阵与输入数据之间进行相乘是否添加偏置
                 qk_scale=None,  # 对qk缩放的比例, 默认为 k 矩阵维度的平方根
                 attn_drop_ratio=0.,  # 注意力层失活率
                 proj_drop_ratio=0.):  # 最后线性映射层失活率
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        # 每个头处理的维度数
        head_dim = embed_dim // num_head
        self.scale = qk_scale or head_dim ** -0.5
        # 将输入数据映射成q, k, v矩阵
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        # qkv() -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape -> [batch_size, num_patches + 1, 3, num_head, embed_dim_per_head]
        # permute -> [3, batch_size, num_head, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
        # [batch_size, num_head, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose -> [batch_size, num_head, embed_dim_per_head, num_patches + 1]
        # @ -> [batch_size, num_head, num_patches + 1, num_patches+1]
        # reshape -> [batch_size, num_patches + 1, total_embed_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @ -> [batch_size, num_head, embed_dim_per_head, num_patches + 1]
        # transpose -> [batch_size, embed_dim_per_head, num_head, num_patches + 1]
        # reshape -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_head,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        """
        Transformer Encoder
        Args:
            embed_dim: encoder输入维度
            num_head: 多头自注意力机制头数量
            mlp_ratio: MLP模块中隐藏层通道扩展比率
            qkv_bias: qkv是否使用偏执
            qk_scale: qk缩放比例
        """
        super(Block, self).__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_head=num_head, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_path_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = FeedForward(in_planes=embed_dim, hidden_planes=hidden_dim, dropout=drop_ratio)

    def forward(self, x):
        output = self.norm1(x)
        output = self.attn(output)
        output = self.drop_path(output)

        x += output

        output = self.norm2(x)
        output = self.mlp(output)
        output = self.drop_path(output)
        x += output
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False,
                 qk_scale=None, drop_ratio=0., attn_drop_ratio=0.,
                 drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        """
        Vision Transformer
        Args:
            image_size: (int, tuple) input image size
            patch_size: (int, tuple) patch size
            in_c: (int) number of input channels
            num_classes: (int) number of classes for classification head
            embed_dim: embedding dimension
            depth: (int) depth of transformer
            num_heads: (int) number of attention heads
            mlp_ratio: (int) ratio of mlp hidden dim to embedding dim
            qkv_bias: (bool) enable bias for qkv if True
            qk_scale: (float) override default qk scale of head_dim ** -0.5
            drop_ratio: (float) dropout rate
            attn_drop_ratio: (float) attention dropout rate
            drop_path_ratio: (float) stochastic depth rate
            embed_layer: (nn.Module) patch embedding layer
            norm_layer: (nn.Module) normalization layer
            act_layer: (nn.Module) activation layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_token = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(image_size=image_size, patch_size=patch_size, in_c=in_c,
                                       embed_dim=embed_dim)
        num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_token, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # 统计深度衰减规律
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(
            *[Block(embed_dim=embed_dim, num_head=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                    norm_layer=norm_layer, act_layer=act_layer) for i in range(depth)]
        )
        self.norm = norm_layer(embed_dim)

        # 分类器
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    model = VisionTransformer(image_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              num_classes=num_classes)
    return model


if __name__ == '__main__':
    model = VisionTransformer()
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    print(y.shape)
