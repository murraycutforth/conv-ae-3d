import unittest
import torch
import torch.nn as nn
from conv_ae_3d.models.baseline_model import ConvAutoencoderBaseline
from conv_ae_3d.models.vae_model import VariationalAutoEncoder3D
from conv_ae_3d.models.efficient_vae_model import EfficientVariationalAutoEncoder3D


class TestConvAutoencoderBaseline(unittest.TestCase):
    def setUp(self):
        self.model = ConvAutoencoderBaseline(
            dim=16,
            dim_mults=(1, 2, 4, 8),
            channels=1,
            z_channels=1,
            block_type=0,
        )
        self.input_tensor = torch.randn(1, 1, 64, 64, 64)

    def test_construction(self):
        self.assertIsInstance(self.model, ConvAutoencoderBaseline)

    def test_forward_pass(self):
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, self.input_tensor.shape)


class TestVariationalAutoEncoder3D_0(unittest.TestCase):
    def setUp(self):
        self.model = VariationalAutoEncoder3D(
            dim=16,
            dim_mults=(1, 2, 4, 8),
            channels=1,
            z_channels=1,
            block_type=0,
        )
        self.input_data = torch.randn(1, 1, 32, 32, 32)  # Batch size of 1, 1 channel, 32x32x32 volume

    def test_initialization(self):
        self.assertIsInstance(self.model, VariationalAutoEncoder3D)

    def test_encode(self):
        posterior = self.model.encode(self.input_data)
        self.assertEqual(posterior.mean.shape, (1, 1, 4, 4, 4))  # Check the shape of the mean
        self.assertEqual(posterior.logvar.shape, (1, 1, 4, 4, 4))  # Check the shape of the logvar

    def test_decode(self):
        z = torch.randn(1, 1, 4, 4, 4)  # Latent variable with the expected shape
        decoded = self.model.decode(z)
        self.assertEqual(decoded.shape, (1, 1, 32, 32, 32))  # Check the shape of the decoded output

    def test_forward(self):
        reconstructed = self.model(self.input_data)
        posterior = self.model.encode(self.input_data)
        self.assertEqual(reconstructed.shape, (1, 1, 32, 32, 32))  # Check the shape of the reconstructed output
        self.assertEqual(posterior.mean.shape, (1, 1, 4, 4, 4))  # Check the shape of the posterior mean


class TestVariationalAutoEncoder3D_1(unittest.TestCase):
    def setUp(self):
        self.model = VariationalAutoEncoder3D(
            dim=10,
            dim_mults=(1, 2, 4, 4, 8),
            channels=1,
            z_channels=1,
            block_type=1,
        )
        self.input_data = torch.randn(1, 1, 32, 32, 32)  # Batch size of 1, 1 channel, 32x32x32 volume

    def test_initialization(self):
        self.assertIsInstance(self.model, VariationalAutoEncoder3D)

    def test_encode(self):
        posterior = self.model.encode(self.input_data)
        self.assertEqual(posterior.mean.shape, (1, 1, 4, 4, 4))  # Check the shape of the mean
        self.assertEqual(posterior.logvar.shape, (1, 1, 4, 4, 4))  # Check the shape of the logvar

    def test_decode(self):
        z = torch.randn(1, 1, 4, 4, 4)  # Latent variable with the expected shape
        decoded = self.model.decode(z)
        self.assertEqual(decoded.shape, (1, 1, 32, 32, 32))  # Check the shape of the decoded output

    def test_forward(self):
        reconstructed = self.model(self.input_data)
        posterior = self.model.encode(self.input_data)
        self.assertEqual(reconstructed.shape, (1, 1, 32, 32, 32))  # Check the shape of the reconstructed output
        self.assertEqual(posterior.mean.shape, (1, 1, 4, 4, 4))  # Check the shape of the posterior mean

class TestEfficientVariationalAutoEncoder(unittest.TestCase):
    def setUp(self):
        self.model = EfficientVariationalAutoEncoder3D(
            dim=8,
            dim_mults=(1, 1, 1, 1),
            channels=1,
            z_channels=1,
            block_type=1,
        )
        self.input_data = torch.randn(1, 1, 32, 32, 32)  # Batch size of 1, 1 channel, 32x32x32 volume

    def test_initialization(self):
        self.assertIsInstance(self.model, EfficientVariationalAutoEncoder3D)

    def test_encode(self):
        posterior = self.model.encode(self.input_data)
        self.assertEqual(posterior.mean.shape, (1, 1, 8, 8, 8))  # Check the shape of the mean
        self.assertEqual(posterior.logvar.shape, (1, 1, 8, 8, 8))  # Check the shape of the logvar

    def test_decode(self):
        z = torch.randn(1, 1, 4, 4, 4)  # Latent variable with the expected shape
        decoded = self.model.decode(z)
        self.assertEqual(decoded.shape, (1, 1, 32, 32, 32))  # Check the shape of the decoded output

    def test_forward(self):
        reconstructed = self.model(self.input_data)
        posterior = self.model.encode(self.input_data)
        self.assertEqual(reconstructed.shape, (1, 1, 32, 32, 32))  # Check the shape of the reconstructed output
        self.assertEqual(posterior.mean.shape, (1, 1, 4, 4, 4))  # Check the shape of the posterior mean


if __name__ == '__main__':
    unittest.main()