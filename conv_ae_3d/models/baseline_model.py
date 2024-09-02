import logging
import typing

import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, norm):
        super().__init__()
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.norm_1 = norm(out_channels)
        self.norm_2 = norm(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.norm_1(x)
        x = self.conv_2(x)
        x = self.activation(x)
        x = self.norm_2(x)
        return x


class ConvBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, output_padding, activation, norm):
        super().__init__()
        self.conv_1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.norm_1 = norm(out_channels)
        self.norm_2 = norm(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.norm_1(x)
        x = self.conv_2(x)
        x = self.activation(x)
        x = self.norm_2(x)
        return x


class FirstConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, norm):
        super().__init__()
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = activation
        self.norm_1 = norm(out_channels)
        self.norm_2 = norm(out_channels)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.norm_1(x)
        x = self.conv_2(x)
        x = self.activation(x)
        x = self.norm_2(x)
        return x


class FinalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, output_padding, activation, norm):
        super().__init__()
        self.conv_1 = nn.ConvTranspose3d(in_channels, out_channels, stride=2, output_padding=output_padding, kernel_size=kernel_size, padding=padding)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv_3 = nn.Conv3d(out_channels, 1, kernel_size=1, padding=0)
        self.activation = activation
        self.norm_1 = norm(out_channels)
        self.norm_2 = norm(out_channels)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.norm_1(x)
        x = self.conv_2(x)
        x = self.activation(x)
        x = self.norm_2(x)
        x = self.conv_3(x)
        return x


class ConvAutoencoderBaseline(nn.Module):
    """Basic 3D conv autoencoder class, uses strided convolutions to downsample inputs, and transposed convolutions to
    upsample them. The bottleneck can be either flat or not.

    We purposefully keep the number of feature maps at the full resolution small to limit GPU memory usage.

    """
    def __init__(self,
                 image_shape: tuple,
                 flat_bottleneck: bool = True,
                 latent_dim: int = 100,
                 activation: nn.Module = nn.ReLU(),
                 norm: typing.Type[nn.Module] = nn.InstanceNorm3d,
                 feat_map_sizes: typing.Iterable = (4, 32, 64, 128),
                 final_activation: typing.Optional[str] = None,
                 ):
        super().__init__()
        self.flat_bottleneck = flat_bottleneck
        self.latent_dim = latent_dim
        self.activation = activation
        self.norm = norm

        self.encoder_outer = nn.Sequential(
            FirstConvBlock(in_channels=1, out_channels=feat_map_sizes[0], kernel_size=3, padding=1, activation=activation, norm=norm),
            *[ConvBlockDown(in_channels, out_channels, kernel_size=3, padding=1, activation=activation, norm=norm) \
              for in_channels, out_channels in zip(feat_map_sizes[:-1], feat_map_sizes[1:])]
        )

        final_shape = [s // (2 ** (len(feat_map_sizes) - 1)) for s in image_shape]

        if flat_bottleneck:
            self.bottleneck_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_map_sizes[-1] * np.prod(final_shape), latent_dim),
            )
            self.encoder = nn.Sequential(
                self.encoder_outer,
                self.bottleneck_encoder,
            )
        else:
            self.encoder = self.encoder_outer

        self.decoder_outer = nn.Sequential(
            *[ConvBlockUp(in_channels, out_channels, kernel_size=3, padding=1, output_padding=1, activation=activation, norm=norm) \
              for in_channels, out_channels in zip(feat_map_sizes[:1:-1], feat_map_sizes[-2:0:-1])],
            FinalConvBlock(in_channels=feat_map_sizes[1], out_channels=feat_map_sizes[0], kernel_size=3, padding=1, output_padding=1, activation=activation, norm=norm)
        )

        if flat_bottleneck:
            self.bottleneck_decoder = nn.Sequential(
                nn.Linear(latent_dim, feat_map_sizes[-1] * np.prod(final_shape)),
                nn.Unflatten(1, (feat_map_sizes[-1], *final_shape)),
            )
            self.decoder = nn.Sequential(
                self.bottleneck_decoder,
                self.decoder_outer,
            )
        else:
            self.decoder = self.decoder_outer

        if final_activation is None:
            self.decoder.add_module('final_activation', nn.Identity())
        elif final_activation == 'sigmoid':
            self.decoder.add_module('final_activation', nn.Sigmoid())
        else:
            raise ValueError(f'Unknown final activation: {final_activation}')

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f'Constructed ConvAutoencoderBaseline with {num_params} parameters')
        logger.debug(f'Model architecture: \n{self}')
        bottleneck_size = latent_dim if flat_bottleneck else np.prod(final_shape) * feat_map_sizes[-1]
        logger.info(f'Input size: {np.prod(image_shape)}, bottleneck size: {bottleneck_size}, compression ratio: {np.prod(image_shape) / bottleneck_size}')

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x
