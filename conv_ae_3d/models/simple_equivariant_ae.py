import torch
import numpy as np
from torch import nn


class SimpleEquivariantAENoDownsamples(nn.Module):
    def __init__(self, z_channels):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1, z_channels, 3, padding=0),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            #nn.ConvTranspose3d(z_channels, 1, 3),
            #nn.Identity(),
            nn.Conv3d(z_channels, 1, 3, padding=0),
        )

        self.z_padder = nn.ZeroPad3d(2)

    def forward(self, x):
        z = self.encoder(x)
        z = self.z_padder(z)
        x_recon = self.decoder(z)
        return x_recon
