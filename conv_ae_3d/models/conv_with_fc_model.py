import torch.nn as nn

from conv_ae_3d.models.baseline_model import ConvAutoencoderBaseline


class ResidualBlock(nn.Module):
    def __init__(self, n_features):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(n_features, n_features)
        self.relu = nn.ReLU()
        self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.fc(x)
        out += identity
        out = self.relu(out)
        return out


class ConvAutoencoderWithFC(ConvAutoencoderBaseline):
    def __init__(self,
                 dim,
                 dim_mults,
                 channels,
                 z_channels,
                 block_type,
                 fc_layers,
                 image_shape):
        super().__init__(dim, dim_mults, channels, z_channels, block_type)

        assert len(image_shape) == 3
        assert fc_layers is not None

        self.flatten = nn.Flatten()

        fc_layers_list = []
        for in_d, out_d in zip(fc_layers[:-1], fc_layers[1:]):
            fc_layers_list.append(nn.Linear(in_d, out_d))
            fc_layers_list.append(nn.ReLU())
            #fc_layers_list.append(ResidualBlock(n_features=out_d))
        fc_layers_list.pop()  # Remove last ReLU
        self.fc_layers = nn.Sequential(*fc_layers_list)

        fc_decoder_layers_list = []
        for in_d, out_d in list(zip(fc_layers[1:], fc_layers[:-1]))[::-1]:
            fc_decoder_layers_list.append(nn.Linear(in_d, out_d))
            fc_decoder_layers_list.append(nn.ReLU())
        self.fc_decoder_layers = nn.Sequential(*fc_decoder_layers_list)

        compressed_shape = [int(i / (2 ** (len(dim_mults) - 2))) for i in image_shape]
        self.unflatten = nn.Unflatten(1, (z_channels, *compressed_shape))

    def encode(self, x):
        z = self.encoder(x)
        z = self.flatten(z)
        z = self.fc_layers(z)
        return z

    def decode(self, z):
        z = self.fc_decoder_layers(z)
        z = self.unflatten(z)
        x = self.decoder(z)
        return x

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return x