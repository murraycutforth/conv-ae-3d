import logging
from functools import partial

import numpy as np
import torch.nn as nn

from conv_ae_3d.models.blocks import Upsample_conv1, Downsample, EquivariantResnetBlock_31, EquivariantResnetBlock_31_up, EquivariantResnetBlock_11, ResnetBlock
from conv_ae_3d.models.vae_model import VariationalAutoEncoder3D
from conv_ae_3d.utils import DiagonalGaussianDistribution

logger = logging.getLogger(__name__)


class EquivariantEncoder3D(nn.Module):
    """Predict posterior distribution q(z|x) from input x

    This should be periodic-2**N shift equivariant, where N is the number of downsamples
    """
    def __init__(self,
                 dim: int,
                 dim_mults: tuple,
                 channels: int,
                 z_channels: int,
                 resnet_block_groups: int = 2,
                 use_norm: bool = False,
                 norm_type: str = 'group',  # 'group', 'batch',
                 use_weight_std_conv: bool = True,
                 act_type: str = 'silu'  # 'silu', 'relu', 'elu'
                 ):
        super().__init__()
        self.channels = channels
        self.init_conv = nn.Conv3d(channels, dim, 1, padding=0)

        dims = list(map(lambda m: dim * m, dim_mults))
        in_out = list(zip(dims[:-1], dims[1:]))
        block_class = partial(EquivariantResnetBlock_31,
                              groups=resnet_block_groups,
                              use_norm=use_norm,
                              norm_type=norm_type,
                              use_weight_std_conv=use_weight_std_conv,
                              act_type=act_type
                              )

        self.downs = nn.ModuleList([])
        self.num_downsamples = len(in_out)
        self.num_conv = [2] * self.num_downsamples

        # Each block_class has one 3x3x3 conv block with reduces the size of each dim by 2
        # The Downsample block reduces the size of each dim by a factor of 2
        # For N downsamples, the output is periodic-2**N shift equivariant
        # How do the Conv3D blocks interact with the downsamples in the equivariance?
        for ind, (dim_in, dim_out) in enumerate(in_out):
            #is_last = ind == len(in_out) - 1
            self.downs.append(block_class(dim_in, dim_in))
            self.downs.append(block_class(dim_in, dim_in))
            self.downs.append(Downsample(dim_in, dim_out))
            #self.downs.append(Downsample(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 1, padding=0))

        mid_dim = dims[-1]
        self.mid_block_1 = EquivariantResnetBlock_11(mid_dim, mid_dim, groups=resnet_block_groups, use_norm=use_norm, norm_type=norm_type, use_weight_std_conv=use_weight_std_conv, act_type=act_type)
        self.mid_block_2 = EquivariantResnetBlock_11(mid_dim, mid_dim, groups=resnet_block_groups, use_norm=use_norm, norm_type=norm_type, use_weight_std_conv=use_weight_std_conv, act_type=act_type)
        self.mid_block_3 = EquivariantResnetBlock_11(mid_dim, mid_dim, groups=resnet_block_groups, use_norm=use_norm, norm_type=norm_type, use_weight_std_conv=use_weight_std_conv, act_type=act_type)
        self.mid_block_4 = EquivariantResnetBlock_11(mid_dim, mid_dim, groups=resnet_block_groups, use_norm=use_norm, norm_type=norm_type, use_weight_std_conv=use_weight_std_conv, act_type=act_type)
        self.final_block = nn.Conv3d(mid_dim, 2 * z_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.init_conv(x)

        for block in self.downs:
            h = block(h)

        h = self.mid_block_1(h)
        h = self.mid_block_2(h)
        h = self.mid_block_3(h)
        h = self.mid_block_4(h)
        h = self.final_block(h)
        return h


class EquivariantDecoder3D(nn.Module):
    """Reconstruct input x from latent variable z

    Modified so that number of upsamples is consistent with number of downsamples in encoder, and can be made
    translation-equivariant.
    """
    def __init__(self,
                 dim: int,
                 dim_mults: tuple,
                 channels: int,
                 z_channels: int,
                 resnet_block_groups: int = 2,
                    use_norm: bool = False,
                 norm_type: str = 'group',  # 'group', 'batch',
                 use_weight_std_conv: bool = True,
                 act_type: str = 'silu',
                 ):
        super().__init__()
        self.channels = channels

        dims = list(map(lambda m: dim * m, dim_mults))
        in_out = list(zip(dims[:-1], dims[1:]))
        block_class = partial(EquivariantResnetBlock_31_up,
                              groups=resnet_block_groups,
                              use_norm=use_norm,
                              norm_type=norm_type,
                              use_weight_std_conv=use_weight_std_conv,
                              act_type=act_type
                              )

        self.ups = nn.ModuleList([])
        self.mid_block_1 = EquivariantResnetBlock_11(z_channels, dims[-1],
                                                     groups=resnet_block_groups,
                                                     use_norm=use_norm,
                                                     norm_type=norm_type,
                                                     use_weight_std_conv=use_weight_std_conv,
                                                     act_type=act_type)
        self.mid_block_2 = EquivariantResnetBlock_11(dims[-1], dims[-1],
                                                     groups=resnet_block_groups,
                                                     use_norm=use_norm,
                                                     norm_type=norm_type,
                                                     use_weight_std_conv=use_weight_std_conv,
                                                     act_type=act_type
                                                     )
        self.mid_block_3 = EquivariantResnetBlock_11(dims[-1], dims[-1],
                                                        groups=resnet_block_groups,
                                                        use_norm=use_norm,
                                                        norm_type=norm_type,
                                                        use_weight_std_conv=use_weight_std_conv,
                                                        act_type=act_type
                                                        )
        self.mid_block_4 = EquivariantResnetBlock_11(dims[-1], dims[-1],
                                                     groups=resnet_block_groups,
                                                     use_norm=use_norm,
                                                     norm_type=norm_type,
                                                     use_weight_std_conv=use_weight_std_conv,
                                                     act_type=act_type
                                                     )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.ups.append(Upsample_conv1(dim_out, dim_out))
            self.ups.append(block_class(dim_out, dim_in))
            self.ups.append(block_class(dim_in, dim_in))

        out_dim = dims[0]
        self.final_block_1 = EquivariantResnetBlock_11(out_dim, out_dim,
                                                       groups=resnet_block_groups,
                                                       use_norm=use_norm,
                                                       norm_type=norm_type,
                                                       use_weight_std_conv=use_weight_std_conv,
                                                       act_type=act_type
                                                       )
        self.final_conv = nn.Conv3d(out_dim, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, z):
        h = self.mid_block_1(z)
        h = self.mid_block_2(h)
        h = self.mid_block_3(h)
        h = self.mid_block_4(h)

        for block in self.ups:
            h = block(h)

        h = self.final_block_1(h)
        h = self.final_conv(h)
        return h



class EquivariantVariationalAutoEncoder3D(VariationalAutoEncoder3D):
    def __init__(self,
                 dim,
                 dim_mults,
                 channels,
                 z_channels,
                 im_shape = None,
                 use_weight_std_conv: bool = True,
                 use_norm: bool = False,
                 norm_type: str = 'group',  # 'group', 'batch'
                 group_norm_size: int = 4,
                 act_type: str = 'silu'
                 ):
        super().__init__(dim=dim, dim_mults=dim_mults, channels=channels, z_channels=z_channels, block_type=1, final_kernel_size=1)
        self.encoder = EquivariantEncoder3D(dim, dim_mults, channels, z_channels,
                                            resnet_block_groups=group_norm_size,
                                            use_norm=use_norm,
                                            norm_type=norm_type,
                                            use_weight_std_conv=use_weight_std_conv,
                                            act_type=act_type)
        self.decoder = EquivariantDecoder3D(dim, dim_mults, channels, z_channels,
                                            resnet_block_groups=group_norm_size,
                                            use_norm=use_norm,
                                            norm_type=norm_type,
                                            use_weight_std_conv=use_weight_std_conv,
                                            act_type=act_type)

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f'Constructed EquivariantVariationAutoEncoder3D with {num_params} parameters')
        logger.debug(f'Model architecture: \n{self}')

        if not im_shape is None:
            final_shape = np.array(im_shape) // (2 ** (len(dim_mults) - 2))
            bottleneck_size = np.prod(final_shape) * z_channels
            logger.info(
                f'Input size: {np.prod(im_shape)}, bottleneck shape: {(z_channels, *final_shape)}, compression ratio: {np.prod(im_shape) / bottleneck_size}')

    @property
    def encoder_shift_equivariance_periodicity(self):
        N = self.encoder.num_downsamples
        return 2**N

    @property
    def encoder_receptive_field_size(self):
        N = self.encoder.num_downsamples
        c = self.encoder.num_conv

        rsize = 1
        for i in range(N):
            rsize = 2 * c[i] + 2 * rsize

        return rsize

    @property
    def encoder_padding_size(self):
        N = self.encoder.num_downsamples
        return self.encoder_receptive_field_size - 2**N