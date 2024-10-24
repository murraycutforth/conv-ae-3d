import logging
from functools import partial

import numpy as np
import torch.nn as nn

from conv_ae_3d.models.blocks import Upsample, Downsample, ResnetBlock, BasicBlock
from conv_ae_3d.utils import DiagonalGaussianDistribution

logger = logging.getLogger(__name__)


class Encoder3D(nn.Module):
    """Predict posterior distribution q(z|x) from input x
    Outputs
    """
    def __init__(self,
                 dim: int,
                 dim_mults: tuple,
                 channels: int,
                 z_channels: int,
                 block_type: int,
                 resnet_block_groups: int = 4
                 ):
        super().__init__()
        self.channels = channels
        self.init_conv = nn.Conv3d(channels, dim, 1, padding=0)

        dims = list(map(lambda m: dim * m, dim_mults))
        in_out = list(zip(dims[:-1], dims[1:]))

        if block_type == 0:
            block_class = BasicBlock
        elif block_type == 1:
            block_class = partial(ResnetBlock, groups=resnet_block_groups)
        else:
            raise ValueError(f"Invalid block type: {block_type}")

        self.downs = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == len(in_out) - 1

            self.downs.append(block_class(dim_in, dim_in))
            self.downs.append(block_class(dim_in, dim_in))
            self.downs.append(Downsample(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 3, padding=1))

        mid_dim = dims[-1]
        self.mid_block_1 = block_class(mid_dim, mid_dim)
        self.mid_block_2 = block_class(mid_dim, mid_dim)
        self.final_block = nn.Conv3d(mid_dim, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.init_conv(x)

        for block in self.downs:
            h = block(h)

        h = self.mid_block_1(h)
        h = self.mid_block_2(h)
        h = self.final_block(h)
        return h


class Decoder3D(nn.Module):
    """Reconstruct input x from latent variable z
    """
    def __init__(self,
                 dim: int,
                 dim_mults: tuple,
                 channels: int,
                 z_channels: int,
                    block_type: int,
                 resnet_block_groups: int = 4
                 ):
        super().__init__()
        self.channels = channels

        dims = list(map(lambda m: dim * m, dim_mults))
        in_out = list(zip(dims[:-1], dims[1:]))

        if block_type == 0:
            block_class = BasicBlock
        elif block_type == 1:
            block_class = partial(ResnetBlock, groups=resnet_block_groups)
        else:
            raise ValueError(f"Invalid block type: {block_type}")

        self.ups = nn.ModuleList([])

        self.init_conv = nn.Conv3d(z_channels, dims[-1], 3, padding=1)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == len(in_out) - 1

            self.ups.append(block_class(dim_out, dim_out))
            self.ups.append(block_class(dim_out, dim_out))
            self.ups.append(Upsample(dim_out, dim_in) if not is_last else nn.Conv3d(dim_out, dim_in, 3, padding=1))

        out_dim = dims[0]
        self.mid_block_1 = block_class(out_dim, out_dim)
        self.mid_block_2 = block_class(out_dim, out_dim)
        self.final_block = nn.Conv3d(out_dim, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.init_conv(z)

        for block in self.ups:
            h = block(h)

        h = self.mid_block_1(h)
        h = self.mid_block_2(h)
        h = self.final_block(h)
        return h


class VariationalAutoEncoder3D(nn.Module):
    def __init__(self,
                 dim,
                 dim_mults,
                 channels,
                 z_channels,
                 block_type,
                 im_shape = None):
        super().__init__()
        self.encoder = Encoder3D(dim, dim_mults, channels, z_channels, block_type=block_type)
        self.decoder = Decoder3D(dim, dim_mults, channels, z_channels, block_type=block_type)

        num_params = sum(p.numel() for p in self.parameters())
        print(f'Constructed VariationAutoEncoder3D with {num_params} parameters')
        logger.debug(f'Model architecture: \n{self}')

        if not im_shape is None:
            final_shape = np.array(im_shape) // (2 ** (len(dim_mults) - 2))
            bottleneck_size = np.prod(final_shape) * z_channels
            print(
                f'Input size: {np.prod(im_shape)}, bottleneck shape: {(z_channels, *final_shape)}, compression ratio: {np.prod(im_shape) / bottleneck_size}')

    def encode(self, x):
        h = self.encoder(x)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=False, return_posterior=False):
        posterior = self.encode(input)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        dec = self.decode(z)

        if return_posterior:
            return dec, posterior
        else:
            return dec