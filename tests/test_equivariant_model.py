import unittest
import torch
import numpy as np
from conv_ae_3d.models.equivariant_vae_model import EquivariantEncoder3D, EquivariantVariationalAutoEncoder3D


class TestEquivariantVariationalAutoEncoder3D(unittest.TestCase):
    def setUp(self):
        self.input_data = torch.randn(1, 1, 100, 100, 100)  # Batch size of 1, 1 channel, 32x32x32 volume

    def test_forward_pass_0downsamples(self):
        vae = EquivariantVariationalAutoEncoder3D(
            dim=4,
            dim_mults=(1,),
            channels=1,
            z_channels=8,
        )
        output = vae(self.input_data[:, :, :30, :30, :30])
        self.assertEqual(vae.encoder.num_downsamples, 0)
        self.assertEqual(output.shape, (1, 1, 30, 30, 30))  # No downsamples, no change in shape

        self.assertEqual(vae.encoder_receptive_field_size, 1)
        self.assertEqual(vae.encoder_padding_size, 0)
        self.assertEqual(vae.encoder_shift_equivariance_periodicity, 1)

    def test_forward_pass_1downsamples(self):
        vae = EquivariantVariationalAutoEncoder3D(
            dim=4,
            dim_mults=(1, 1),
            channels=1,
            z_channels=8,
        )
        self.assertEqual(vae.encoder_receptive_field_size, 6)
        self.assertEqual(vae.encoder_padding_size, 4)
        self.assertEqual(vae.encoder_shift_equivariance_periodicity, 2)
        self.assertEqual(vae.encoder.num_downsamples, 1)

        output = vae(self.input_data[:, :, :30, :30, :30])
        self.assertEqual(output.shape, (1, 1, 30, 30, 30))  # 1 downsample, no change in shape

    def test_forward_pass_1downsamples_1(self):
        vae = EquivariantVariationalAutoEncoder3D(
            dim=4,
            dim_mults=(1, 2),
            channels=1,
            z_channels=8,
        )
        self.assertEqual(vae.encoder_receptive_field_size, 6)
        self.assertEqual(vae.encoder_padding_size, 4)
        self.assertEqual(vae.encoder_shift_equivariance_periodicity, 2)
        self.assertEqual(vae.encoder.num_downsamples, 1)

        output = vae(self.input_data[:, :, :30, :30, :30])
        self.assertEqual(output.shape, (1, 1, 30, 30, 30))  # 1 downsample, no change in shape

    def test_forward_pass_2downsamples(self):
        vae = EquivariantVariationalAutoEncoder3D(
            dim=4,
            dim_mults=(1, 1, 1),
            channels=1,
            z_channels=8,
        )
        self.assertEqual(vae.encoder_receptive_field_size, 16)
        self.assertEqual(vae.encoder_padding_size, 12)
        self.assertEqual(vae.encoder_shift_equivariance_periodicity, 4)
        self.assertEqual(vae.encoder.num_downsamples, 2)

        output = vae(self.input_data[:, :, :32, :32, :32])
        self.assertEqual(output.shape, (1, 1, 32, 32, 32))

    def test_forward_pass_3downsamples(self):
        vae = EquivariantVariationalAutoEncoder3D(
            dim=4,
            dim_mults=(1, 1, 1, 1),
            channels=1,
            z_channels=8,
        )
        self.assertEqual(vae.encoder_receptive_field_size, 36)
        self.assertEqual(vae.encoder_padding_size, 28)
        self.assertEqual(vae.encoder_shift_equivariance_periodicity, 8)
        self.assertEqual(vae.encoder.num_downsamples, 3)

        output = vae(self.input_data[:, :, :36, :36, :36])
        self.assertEqual(output.shape, (1, 1, 36, 36, 36))


class TestEquivariantEncoder3D(unittest.TestCase):
    def setUp(self):
        self.input_data = torch.randn(1, 1, 100, 100, 100)  # Batch size of 1, 1 channel, 32x32x32 volume

    def test_forward_pass_0downsamples(self):
        encoder = EquivariantEncoder3D(
            dim=4,
            dim_mults=(1,),
            channels=1,
            z_channels=8,
            resnet_block_groups=2
        )
        output = encoder(self.input_data[:, :, :30, :30, :30])
        self.assertEqual(encoder.num_downsamples, 0)
        self.assertEqual(output.shape, (1, 16, 30, 30, 30))  # 2 conv3 layers - lose 2 pixels on each side

    def test_forward_pass_1downsamples(self):
        encoder = EquivariantEncoder3D(
            dim=4,
            dim_mults=(1, 1),
            channels=1,
            z_channels=8,
            resnet_block_groups=2
        )
        output = encoder(self.input_data[:, :, :30, :30, :30])
        self.assertEqual(encoder.num_downsamples, 1)
        self.assertEqual(output.shape, (1, 16, 13, 13, 13)) # 2 conv3 layers, 1 downsample: N -> (N - 4) / 2

    def test_forward_pass_1downsamples_1(self):
        # Check forward pass still works with changing dim_mults
        encoder = EquivariantEncoder3D(
            dim=24,
            dim_mults=(1, 2),
            channels=1,
            z_channels=8,
        )
        output = encoder(self.input_data[:, :, :30, :30, :30])
        self.assertEqual(output.shape, (1, 16, 13, 13, 13))  # 2 conv3 layers, 1 downsample: N -> (N - 4) / 2

    def test_forward_pass_2downsamples(self):
        encoder = EquivariantEncoder3D(
            dim=4,
            dim_mults=(1, 1, 1),
            channels=1,
            z_channels=8,
            resnet_block_groups=2
        )
        output = encoder(self.input_data[:, :, :28, :28, :28])  # Should be periodic-4 shift invariant
        self.assertEqual(encoder.num_downsamples, 2)
        self.assertEqual(output.shape, (1, 16, 4, 4, 4))  # 2 x (2 conv3, downsample). x = ((N - 4) / 2 - 4) / 2, or N = (2x + 4) * 2 + 4. x=1, N=16

    def test_forward_pass_3downsamples(self):
        encoder = EquivariantEncoder3D(
            dim=4,
            dim_mults=(1, 1, 1, 1),
            channels=1,
            z_channels=8,
            resnet_block_groups=2
        )
        output = encoder(self.input_data[:, :, :36, :36, :36])
        self.assertEqual(encoder.num_downsamples, 3)
        self.assertEqual(output.shape, (1, 16, 1, 1, 1))  # 3 x (2 conv3, downsample). x = (((N - 4) / 2 - 4) / 2 - 4) / 2, or N = ((2x + 4) * 2 + 4) * 2 + 4. x=1, N=36

        output = encoder(self.input_data[:, :, :68, :68, :68])
        self.assertEqual(encoder.num_downsamples, 3)
        self.assertEqual(output.shape, (1, 16, 5, 5, 5))  # 3 x (2 conv3, downsample). x = (((N - 4) / 2 - 4) / 2 - 4) / 2, or N = ((2x + 4) * 2 + 4) * 2 + 4. x=1, N=36

    def test_shift_equivariance_0downsamples_x(self):
        periodicity = 1
        window_size = 2

        encoder = EquivariantEncoder3D(
            dim=4,
            dim_mults=(1,),
            channels=1,
            z_channels=1,
            resnet_block_groups=2,
            use_norm=False
        )

        # This means that shifting the input in any direction with a step of `periodicity` should shift the output by 1

        # We have shift in x-dim only
        z_1 = encoder(self.input_data[:, :, :window_size, :window_size, :window_size])
        z_2 = encoder(self.input_data[:, :, periodicity: window_size+periodicity, :window_size, :window_size])

        self.assertTrue(torch.allclose(z_1[:, :, 1:, :, :], z_2[:, :, :-1, :, :], atol=1e-6),
                        "The encoder output is not shift equivariant")

    def test_shift_equivariance_0downsamples_usenorm_x(self):
        periodicity = 1
        window_size = 2

        encoder = EquivariantEncoder3D(
            dim=4,
            dim_mults=(1,),
            channels=1,
            z_channels=1,
            resnet_block_groups=2,
            use_norm=True  # This should break shift equivariance
        )

        # This means that shifting the input in any direction with a step of `periodicity` should shift the output by 1

        # We have shift in x-dim only
        z_1 = encoder(self.input_data[:, :, :window_size, :window_size, :window_size])
        z_2 = encoder(self.input_data[:, :, periodicity: window_size+periodicity, :window_size, :window_size])

        # Now z2[:, :, :-1, :, :] should be the same as z1[:, :, 1:, :, :]
        self.assertFalse(torch.allclose(z_1[:, :, 1:, :, :], z_2[:, :, :-1, :, :], atol=1e-6),
                        "The encoder output is equivariant with norm on?")

    def test_shift_equivariance_1downsamples_x(self):
        periodicity = 2
        window_size = 8

        encoder = EquivariantEncoder3D(
            dim=4,
            dim_mults=(1, 1),
            channels=1,
            z_channels=1,
            resnet_block_groups=2,
            use_norm=False
        )

        # This means that shifting the input in any direction with a step of `periodicity` should shift the output by 1

        # We have shift in x-dim only
        z_1 = encoder(self.input_data[:, :, :window_size, :window_size, :window_size])
        z_2 = encoder(self.input_data[:, :, periodicity: window_size+periodicity, :window_size, :window_size])

        # Now z2[:, :, :-1, :, :] should be the same as z1[:, :, 1:, :, :]
        self.assertTrue(torch.allclose(z_1[:, :, 1:, :, :], z_2[:, :, :-1, :, :], atol=1e-6),
                        "The encoder output is not shift equivariant")

    def test_shift_equivariance_2downsamples_x(self):
        periodicity = 4
        window_size = 16

        encoder = EquivariantEncoder3D(
            dim=4,
            dim_mults=(1, 1, 1),
            channels=1,
            z_channels=1,
            resnet_block_groups=2,
            use_norm=False
        )

        # This means that shifting the input in any direction with a step of `periodicity` should shift the output by 1

        # We have shift in x-dim only
        z_1 = encoder(self.input_data[:, :, :window_size, :window_size, :window_size])
        z_2 = encoder(self.input_data[:, :, periodicity: window_size+periodicity, :window_size, :window_size])

        # Now z2[:, :, :-1, :, :] should be the same as z1[:, :, 1:, :, :]
        self.assertTrue(torch.allclose(z_1[:, :, 1:, :, :], z_2[:, :, :-1, :, :], atol=1e-6),
                        "The encoder output is not shift equivariant")

    def test_shift_equivariance_3downsamples_x(self):

        encoder = EquivariantEncoder3D(
            dim=4,
            dim_mults=(1, 1, 1, 1),
            channels=1,
            z_channels=1,
            use_norm=False
        )

        # This means that shifting the input in any direction with a step of `periodicity` should shift the output by 1

        periodicity = 8
        window_size = 52

        z_1 = encoder(self.input_data[:, :, :window_size, :window_size, :window_size])

        # Shift in x-dim only
        z_2 = encoder(self.input_data[:, :, periodicity: window_size+periodicity, :window_size, :window_size])

        self.assertTrue(torch.allclose(z_1[:, :, 1:, :, :], z_2[:, :, :-1, :, :], atol=1e-6),
                        "The encoder output is not shift equivariant")

    def test_shift_equivariance_3downsamples_xyz(self):
        periodicity = 8
        window_size = 52

        encoder = EquivariantEncoder3D(
            dim=4,
            dim_mults=(1, 1, 1, 1),
            channels=1,
            z_channels=1,
            use_norm=False
        )

        # This means that shifting the input in any direction with a step of `periodicity` should shift the output by 1

        # We have shift in x-dim only
        z_1 = encoder(self.input_data[:, :, :window_size, :window_size, :window_size])
        z_2 = encoder(self.input_data[:, :, periodicity: window_size+periodicity,
                      periodicity:window_size+periodicity, periodicity:window_size+periodicity])

        # Now z2[:, :, :-1, :, :] should be the same as z1[:, :, 1:, :, :]
        self.assertTrue(torch.allclose(z_1[:, :, 1:, 1:, 1:], z_2[:, :, :-1, :-1, :-1], atol=1e-6),
                        "The encoder output is not shift equivariant")

if __name__ == '__main__':
    unittest.main()
