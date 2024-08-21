import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        """
        Image to Patch Embedding
        Args:
            img_size: (int)输入图像尺寸, 默认为224
            patch_size: (int)patch块大小, 默认为4
            in_chans: (int)输入图像通道数, 默认为3
            embed_dim: (int)线性嵌入层输出通道数, 默认96
            norm_layer: (int)正则化层
        """
        super(PatchEmbed, self).__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        # patch分辨率
        self.patch_resolution = (img_size // patch_size, img_size // patch_size)
        # patch块数量
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]} * {self.img_size[1]})."
        x = self.proj(x)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: tuple[int], dim: int, norm_layer=nn.LayerNorm):
        """
        Patch Merging Layer.
        Args:
            input_resolution: Resolution of input feature.
            dim: Number of input channels.
            norm_layer: Normalization layer. Default: nn.LayerNorm
        """
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.linear = nn.Linear(in_features=4 * dim, out_features=2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        # 对原始特征图进行分组
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        # 在通道维度进行拼接
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B, H/2*W/2 4*C
        x = self.norm(x)
        x = self.linear(x)  # B, H/2*W/2 2*C
        return x

