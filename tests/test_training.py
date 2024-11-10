import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from conv_ae_3d.trainer import MyAETrainer
from conv_ae_3d.models.baseline_model import ConvAutoencoderBaseline
from conv_ae_3d.models.vae_model import VariationalAutoEncoder3D
from conv_ae_3d.metrics import MetricType


class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def unnormalise_array(self, array):
        return array


class TestTrainingNoiseVAE_0(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data, just noise
        self.train_ds = TestDataset(torch.randn(1, 1, 32, 32, 32))

        self.model = VariationalAutoEncoder3D(
            dim=16,
            dim_mults=(1, 2, 2, 2),
            channels=1,
            z_channels=4,
            block_type=0,
            im_shape=(32, 32, 32),
            final_kernel_size=1,
        )

        self.trainer = MyAETrainer(
            model=self.model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=1,
            train_lr=1e-3,
            train_num_epochs=200,
            save_and_sample_every=500,
            results_folder='test_output_noise_vae_0',
            cpu_only=True,
            num_dl_workers=0,
            kl_weight=1e-6,
            sample_posterior=True,
        )

    def test_z_shape(self):
        posterior = self.model.encode(self.train_ds.data)
        z = posterior.mode()
        self.assertEqual(z.shape, (1, 4, 8, 8, 8))

    def test_inference_shape(self):
        for pred, data in self.trainer.run_inference(self.trainer.dl_val, max_n_batches=None):
            self.assertEqual(pred.shape, data.shape)
            self.assertEqual(pred.shape, (32, 32, 32))

    def test_training(self):
        self.trainer.train()
        results = self.trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.3)
        self.assertLess(results['MSE'], 0.2)


class TestTrainingNoiseVAE_1(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data, just noise
        self.train_ds = TestDataset(torch.randn(1, 1, 32, 32, 32))

        self.model = VariationalAutoEncoder3D(
            dim=16,
            dim_mults=(1, 2, 2, 2),
            channels=1,
            z_channels=4,
            block_type=1,
            im_shape=(32, 32, 32)
        )

        self.trainer = MyAETrainer(
            model=self.model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=1,
            train_lr=1e-3,
            train_num_epochs=500,
            save_and_sample_every=500,
            results_folder='test_output_noise_vae_1',
            cpu_only=True,
            num_dl_workers=0,
            kl_weight=1e-6,
        )

    def test_z_shape(self):
        posterior = self.model.encode(self.train_ds)
        z = posterior.mode()
        self.assertEqual(z.shape, (1, 4, 8, 8, 8))

    def test_inference_shape(self):
        for pred, data in self.trainer.run_inference(self.trainer.dl_val, max_n_batches=None):
            self.assertEqual(pred.shape, data.shape)
            self.assertEqual(pred.shape, (32, 32, 32))

    def test_training(self):
        self.trainer.train()
        results = self.trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.3)
        self.assertLess(results['MSE'], 0.2)


class TestTrainingNoiseAEBaseline_0(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data, just noise
        self.train_ds = torch.randn(1, 1, 32, 32, 32)
        self.train_ds = TestDataset(torch.randn(1, 1, 32, 32, 32))

        #self.model = ConvAutoencoderBaseline(
        #    image_shape=(32, 32, 32),
        #    feat_map_sizes=(16, 32, 64)
        #)
        self.model = ConvAutoencoderBaseline(
            dim=16,
            dim_mults=(1, 2, 2, 2),
            channels=1,
            z_channels=4,
            block_type=0,
            im_shape=(32, 32, 32)
        )

        self.trainer = MyAETrainer(
            model=self.model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=1,
            train_lr=1e-3,
            train_num_epochs=200,
            save_and_sample_every=500,
            results_folder='test_output_noise_baseline_0',
            cpu_only=True,
            num_dl_workers=0,
        )

    def test_inference_shape(self):
        for pred, data in self.trainer.run_inference(self.trainer.dl_val, max_n_batches=None):
            self.assertEqual(pred.shape, data.shape)
            self.assertEqual(pred.shape, (32, 32, 32))

    def test_training(self):
        self.trainer.train()
        results = self.trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.3)
        self.assertLess(results['MSE'], 0.2)


class TestTrainingNoiseAEBaseline_1(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data, just noise
        self.train_ds = torch.randn(1, 1, 32, 32, 32)
        self.train_ds = TestDataset(torch.randn(1, 1, 32, 32, 32))

        #self.model = ConvAutoencoderBaseline(
        #    image_shape=(32, 32, 32),
        #    feat_map_sizes=(16, 32, 64)
        #)
        self.model = ConvAutoencoderBaseline(
            dim=16,
            dim_mults=(1, 2, 2, 2),
            channels=1,
            z_channels=4,
            block_type=1,
            im_shape=(32, 32, 32)
        )

        self.trainer = MyAETrainer(
            model=self.model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=1,
            train_lr=1e-3,
            train_num_epochs=1000,
            save_and_sample_every=200,
            results_folder='test_output_noise_baseline_1',
            #cpu_only=True,
            num_dl_workers=0,
        )

    def test_inference_shape(self):
        for pred, data in self.trainer.run_inference(self.trainer.dl_val, max_n_batches=None):
            self.assertEqual(pred.shape, data.shape)
            self.assertEqual(pred.shape, (32, 32, 32))

    def test_training(self):
        self.trainer.train()
        results = self.trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.3)
        self.assertLess(results['MSE'], 0.2)


class TestTrainingSquares(unittest.TestCase):
    def setUp(self):
        self.b = 8
        data = torch.zeros(self.b, 1, 16, 16, 16)
        for i in range(self.b):
            x = torch.randint(2, 10, (1,))
            y = torch.randint(2, 10, (1,))
            z = torch.randint(2, 10, (1,))
            data[i, 0, x:x+5, y:y+5, z:z+5] = 1
        self.train_ds = TestDataset(data)

    def test_baseline_block_0(self):
        model = ConvAutoencoderBaseline(
            dim=24,
            dim_mults=(1, 2, 2, 2),
            channels=1,
            z_channels=4,
            block_type=0,
            im_shape=(16, 16, 16)
        )

        trainer = MyAETrainer(
            model=model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=self.b,
            train_lr=1e-3,
            train_num_epochs=100,
            save_and_sample_every=25,
            results_folder='test_output_squares_baseline_0',
            cpu_only=True,
            num_dl_workers=0,
        )

        trainer.train()
        results = trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.1)
        self.assertLess(results['MSE'], 0.01)

    def test_baseline_block_1(self):
        model = ConvAutoencoderBaseline(
            dim=16,
            dim_mults=(1, 2, 2, 2),
            channels=1,
            z_channels=4,
            block_type=1,
            im_shape=(32, 32, 32)
        )

        trainer = MyAETrainer(
            model=model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=1,
            train_lr=1e-3,
            train_num_epochs=100,
            save_and_sample_every=500,
            results_folder='test_output_squares_baseline_1',
            cpu_only=True,
            num_dl_workers=0,
        )

        trainer.train()
        results = trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.1)
        self.assertLess(results['MSE'], 0.01)

    def test_vae_block_1(self):
        model = VariationalAutoEncoder3D(
            dim=24,
            dim_mults=(1, 2, 2, 2),
            channels=1,
            z_channels=4,
            block_type=1,
            im_shape=(16, 16, 16),
            group_norm_size=2,
            final_kernel_size=1,
        )

        trainer = MyAETrainer(
            model=model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=self.b,
            train_lr=1e-3,
            train_num_epochs=100,
            save_and_sample_every=25,
            results_folder='test_output_squares_vae_0',
            cpu_only=True,
            num_dl_workers=0,
            kl_weight=1e-6,
            sample_posterior=False,
        )

        trainer.train()
        results = trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.1)
        self.assertLess(results['MSE'], 0.01)

    def test_vae_block_0(self):
        model = VariationalAutoEncoder3D(
            dim=24,
            dim_mults=(1, 2, 2, 2),
            channels=1,
            z_channels=4,
            block_type=0,
            im_shape=(16, 16, 16)
        )

        trainer = MyAETrainer(
            model=model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=self.b,
            train_lr=1e-3,
            train_num_epochs=100,
            save_and_sample_every=25,
            results_folder='test_output_squares_vae_0',
            cpu_only=True,
            num_dl_workers=0,
            kl_weight=1e-6,
            sample_posterior=False,
        )

        trainer.train()
        results = trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.1)
        self.assertLess(results['MSE'], 0.01)


class TestTrainingRealData(unittest.TestCase):
    def setUp(self):
        data_dir = Path('test_data') / '51_0.npz'
        data = np.load(data_dir)['rho']
        image_shape = (32, 32, 32)  # For speed
        data = data[image_shape[0]:2 * image_shape[0], image_shape[1]:2 * image_shape[1], image_shape[2]:2 * image_shape[2]]

        # This is the current normalisation
        # Problem - MAE and MSE are not scale invariant
        data = data / 3.0

        data_min = data.min()
        data_max = data.max()

        print(f'Min: {data_min}, Max: {data_max}')

        data = (data - data_min) / (data_max - data_min)

        print(f'Min: {data.min()}, Max: {data.max()}')


        data = torch.tensor(data).unsqueeze(0).unsqueeze(0).float()
        self.train_ds = TestDataset(data)

    def test_baseline_block_0(self):
        model = ConvAutoencoderBaseline(
            dim=32,
            dim_mults=(1, 2, 2),
            channels=1,
            z_channels=1,
            block_type=0,
        )

        trainer = MyAETrainer(
            model=model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=1,
            train_lr=1e-3,
            train_num_epochs=100,
            save_and_sample_every=50,
            results_folder='test_output_realdata_baseline_0',
            cpu_only=True,
            num_dl_workers=0,
            #loss=nn.L1Loss(),
            metric_types=[MetricType.MAE, MetricType.MSE, MetricType.LINF]
        )

        trainer.train()
        results = trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.2)
        self.assertLess(results['MSE'], 0.05)
        self.assertLess(results['LINF'], 1)

    def test_vae_block_1(self):
        model = VariationalAutoEncoder3D(
            dim=32,
            dim_mults=(1, 2, 2),
            channels=1,
            z_channels=1,
            block_type=1,
            group_norm_size=2,
            final_kernel_size=1,
        )

        trainer = MyAETrainer(
            model=model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=1,
            train_lr=1e-4,
            train_num_epochs=100,
            save_and_sample_every=20,
            results_folder='test_output_realdata_vae_1',
            cpu_only=True,
            num_dl_workers=0,
            kl_weight=1e-6,
            sample_posterior=False,
            #loss=nn.L1Loss(),
            metric_types=[MetricType.MAE, MetricType.MSE, MetricType.LINF]
        )

        trainer.train()
        results = trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.2)
        self.assertLess(results['MSE'], 0.05)
        self.assertLess(results['LINF'], 1)

    def test_vae_block_0(self):
        model = VariationalAutoEncoder3D(
            dim=32,
            dim_mults=(1, 2, 2),
            channels=1,
            z_channels=1,
            block_type=0,
        )

        trainer = MyAETrainer(
            model=model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=1,
            train_lr=1e-4,
            train_num_epochs=100,
            save_and_sample_every=20,
            results_folder='test_output_realdata_vae_0',
            cpu_only=True,
            num_dl_workers=0,
            kl_weight=1e-6,
            sample_posterior=False,
            #loss=nn.L1Loss(),
            metric_types=[MetricType.MAE, MetricType.MSE, MetricType.LINF]
        )

        trainer.train()
        results = trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.2)
        self.assertLess(results['MSE'], 0.05)
        self.assertLess(results['LINF'], 1)


if __name__ == '__main__':
    unittest.main()