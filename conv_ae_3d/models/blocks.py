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


def Upsample_conv1(dim, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv3d(dim, dim_out, 1, padding=0),
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


class BlockType2(nn.Module):
    """We set padding to 0, to use this block in a translation-equivariant encoder
    TODO: GroupNorm averages across all pixels and groups of channels. This breaks translation equivariance,
    because the mean and variance are computed across all pixels, so will change as the input window moves over the data.
    """
    def __init__(self, dim, dim_out, use_norm=True, kernel_size=3, padding=0, groups=8):
        super().__init__()
        self.use_norm = use_norm
        self.proj = WeightStandardizedConv3d(dim, dim_out, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        if self.use_norm:
            x = self.norm(x)
        x = self.act(x)
        return x


class ConvBlock(nn.Module):
    """A basic 3D conv + norm + activation block, with norm and activation provided as arguments
    """
    def __init__(self, dim, dim_out, kernel_size, padding, norm, act):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim_out, kernel_size, padding=padding)
        self.norm = norm
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class WeightStandardizedConvBlock(nn.Module):
    """A basic 3D conv + norm + activation block, with weight-standardized conv
    """
    def __init__(self, dim, dim_out, kernel_size, padding, norm, act):
        super().__init__()
        self.conv = WeightStandardizedConv3d(dim, dim_out, kernel_size, padding=padding)
        self.norm = norm
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def construct_norm(norm_type, groups, dim_out):
    if norm_type == 'group':
        return nn.GroupNorm(groups, dim_out)
    elif norm_type == 'batch':
        return nn.BatchNorm3d(dim_out)
    else:
        raise ValueError(f"Invalid norm type: {norm_type}")


def construct_act(act_type):
    if act_type == 'silu':
        return nn.SiLU()
    elif act_type == 'elu':
        return nn.ELU()
    elif act_type == 'relu':
        return nn.ReLU()
    else:
        raise ValueError(f"Invalid activation type: {act_type}")


class EquivariantResnetBlock_31(nn.Module):
    """As below, but modified to be equivariant
    """

    def __init__(self, dim, dim_out, groups=8, use_norm=True, norm_type='group', use_weight_std_conv=True, act_type='silu'):
        super().__init__()

        if use_norm:
            norm1 = construct_norm(norm_type, groups, dim_out)
            norm2 = construct_norm(norm_type, groups, dim_out)
        else:
            norm1 = nn.Identity()
            norm2 = nn.Identity()

        if use_weight_std_conv:
            self.block1 = WeightStandardizedConvBlock(dim, dim_out, kernel_size=3, padding=0, norm=norm1, act=construct_act(act_type))
            self.block2 = WeightStandardizedConvBlock(dim_out, dim_out, kernel_size=1, padding=0, norm=norm2, act=construct_act(act_type))
        else:
            self.block1 = ConvBlock(dim, dim_out, kernel_size=3, padding=0, norm=norm1, act=construct_act(act_type))
            self.block2 = ConvBlock(dim_out, dim_out, kernel_size=1, padding=0, norm=norm2, act=construct_act(act_type))
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)[:, :, 1:-1, 1:-1, 1:-1]


class EquivariantResnetBlock_31_up(nn.Module):
    """As above, but with additional padding, to be used in an upsampling path. This should exactly cancel out the
    size changes from the downsampling path.
    """
    def __init__(self, dim, dim_out, groups=8, use_norm=True, norm_type='group', use_weight_std_conv=True, act_type='silu'):
        super().__init__()

        if use_norm:
            norm1 = construct_norm(norm_type, groups, dim_out)
            norm2 = construct_norm(norm_type, groups, dim_out)
        else:
            norm1 = nn.Identity()
            norm2 = nn.Identity()

        if use_weight_std_conv:
            self.block1 = WeightStandardizedConvBlock(dim, dim_out, kernel_size=3, padding=2, norm=norm1, act=construct_act(act_type))
            self.block2 = WeightStandardizedConvBlock(dim_out, dim_out, kernel_size=1, padding=0, norm=norm2, act=construct_act(act_type))
        else:
            self.block1 = ConvBlock(dim, dim_out, kernel_size=3, padding=2, norm=norm1, act=construct_act(act_type))
            self.block2 = ConvBlock(dim_out, dim_out, kernel_size=1, padding=0, norm=norm2, act=construct_act(act_type))

        self.res_conv = nn.Conv3d(dim, dim_out, kernel_size=1, padding=0) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + F.pad(self.res_conv(x), (1, 1, 1, 1, 1, 1), "constant", 0)




class EquivariantResnetBlock_11(nn.Module):
    """As below, but modified to be equivariant
    """
    def __init__(self, dim, dim_out, groups=8, use_norm=True, norm_type='group', use_weight_std_conv=True, act_type='silu'):
        super().__init__()

        if use_norm:
            norm_1 = construct_norm(norm_type, groups, dim_out)
            norm_2 = construct_norm(norm_type, groups, dim_out)
        else:
            norm_1 = nn.Identity()
            norm_2 = nn.Identity()

        if use_weight_std_conv:
            self.block1 = WeightStandardizedConvBlock(dim, dim_out, kernel_size=1, padding=0, norm=norm_1, act=construct_act(act_type))
            self.block2 = WeightStandardizedConvBlock(dim_out, dim_out, kernel_size=1, padding=0, norm=norm_2, act=construct_act(act_type))
        else:
            self.block1 = ConvBlock(dim, dim_out, kernel_size=1, padding=0, norm=norm_1, act=construct_act(act_type))
            self.block2 = ConvBlock(dim_out, dim_out, kernel_size=1, padding=0, norm=norm_2, act=construct_act(act_type))

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)



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
