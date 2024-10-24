from functools import partial

import torch
from einops import reduce
from einops.layers.torch import Rearrange
from torch import nn as nn
from torch.nn import functional as F


def Upsample(dim, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv3d(dim, dim_out, 3, padding=1),
    )


def Downsample(dim, dim_out):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) (d p3) -> b (c p1 p2 p3) h w d", p1=2, p2=2, p3=2),
        nn.Conv3d(dim * 8, dim_out, 1),
    )


class WeightStandardizedConv3d(nn.Conv3d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) / (var + eps).rsqrt()

        return F.conv3d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class BlockType0(nn.Module):
    """A basic 3D conv + ELU + InstanceNorm block
    """
    def __init__(self, dim, dim_out):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.InstanceNorm3d(dim_out)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class BlockType1(nn.Module):
    """A fancier 3D conv + norm + activation block
    """
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv3d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """Two blocks with a residual connection to input
    See: https://arxiv.org/abs/1512.03385
    """

    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block1 = BlockType1(dim, dim_out, groups=groups)
        self.block2 = BlockType1(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class BasicBlock(nn.Module):
    """Two basic blocks in series
    """
    def __init__(self, dim, dim_out):
        super().__init__()
        self.block1 = BlockType0(dim, dim_out)
        self.block2 = BlockType0(dim_out, dim_out)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h
