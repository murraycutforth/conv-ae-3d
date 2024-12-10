import unittest
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from conv_ae_3d.trainer_ae import MyAETrainer
from conv_ae_3d.trainer_vae import MyVAETrainer
from conv_ae_3d.models.baseline_model import ConvAutoencoderBaseline
from conv_ae_3d.models.vae_model import VariationalAutoEncoder3D
from conv_ae_3d.models.equivariant_vae_model import EquivariantVariationalAutoEncoder3D
from conv_ae_3d.metrics import MetricType


models_to_compare = [
    partial(VariationalAutoEncoder3D, dim=16, dim_mults=(1, 2, 4, 4), channels=1, z_channels=4, block_type=1),
    partial(EquivariantVariationalAutoEncoder3D, dim=16, dim_mults=(1, 2, 4), channels=1, z_channels=4, use_weight_std_conv=True, use_norm=True, norm_type='group', act_type='silu'),
    partial(EquivariantVariationalAutoEncoder3D, dim=16, dim_mults=(1, 2, 4), channels=1, z_channels=4, use_weight_std_conv=True, use_norm=True, norm_type='group', act_type='relu'),
    partial(EquivariantVariationalAutoEncoder3D, dim=16, dim_mults=(1, 2, 4), channels=1, z_channels=4, use_weight_std_conv=True, use_norm=False, act_type='silu'),
    partial(EquivariantVariationalAutoEncoder3D, dim=16, dim_mults=(1, 2, 4), channels=1, z_channels=4, use_weight_std_conv=True, use_norm=False, act_type='relu'),
    partial(EquivariantVariationalAutoEncoder3D, dim=16, dim_mults=(1, 2, 4), channels=1, z_channels=4, use_weight_std_conv=False, use_norm=True, norm_type='group', act_type='silu'),
    partial(EquivariantVariationalAutoEncoder3D, dim=16, dim_mults=(1, 2, 4), channels=1, z_channels=4, use_weight_std_conv=False, use_norm=True, norm_type='group', act_type='relu'),
    partial(EquivariantVariationalAutoEncoder3D, dim=16, dim_mults=(1, 2, 4), channels=1, z_channels=4, use_weight_std_conv=False, use_norm=False, act_type='silu'),
    partial(EquivariantVariationalAutoEncoder3D, dim=16, dim_mults=(1, 2, 4), channels=1, z_channels=4, use_weight_std_conv=False, use_norm=False, act_type='relu'),
]

model_names = [
    'Baseline_VAE',
    'Equivariant_VAE_weightstd_groupnorm_silu',
    'Equivariant_VAE_weightstd_groupnorm_relu',
    'Equivariant_VAE_weightstd_no_norm_silu',
    'Equivariant_VAE_weightstd_no_norm_relu',
    'Equivariant_VAE_no_weightstd_groupnorm_silu',
    'Equivariant_VAE_no_weightstd_groupnorm_relu',
    'Equivariant_VAE_no_weightstd_no_norm_silu',
    'Equivariant_VAE_no_weightstd_no_norm_relu',
]


class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def unnormalise_array(self, array):
        return array


class TestTrainingWhiteNoiseDataset(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data, just noise
        self.train_ds = TestDataset(torch.randn(1, 1, 20, 20, 20))

    def test_all_models(self):
        for model_fn, model_name in zip(models_to_compare, model_names):
            model = model_fn(im_shape=(20, 20, 20))
            trainer = MyVAETrainer(
                model=model,
                dataset_train=self.train_ds,
                dataset_val=self.train_ds,
                train_batch_size=1,
                train_lr=5e-4,
                train_num_epochs=1000,
                save_and_sample_every=5000,
                results_folder=f'test_output/comparison/whitenoise/{model_name}',
                cpu_only=True,
                num_dl_workers=0,
                kl_weight=1e-6,
            )

            trainer.train()

    def test_baseline_block_0(self):
        model = ConvAutoencoderBaseline(
            dim=16,
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
            train_batch_size=1,
            train_lr=1e-3,
            train_num_epochs=200,
            save_and_sample_every=500,
            results_folder='test_output_noise_baseline_0',
            cpu_only=True,
            num_dl_workers=0,
        )

        trainer.train()
        results = trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.1)
        self.assertLess(results['MSE'], 0.01)

    def test_vae_block_1(self):
        model = VariationalAutoEncoder3D(
            dim=16,
            dim_mults=(1, 2, 2, 2),
            channels=1,
            z_channels=4,
            block_type=1,
            im_shape=(16, 16, 16)
        )

        trainer = MyVAETrainer(
            model=model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=1,
            train_lr=1e-3,
            train_num_epochs=200,
            save_and_sample_every=500,
            results_folder='test_output_noise_vae_1',
            cpu_only=True,
            num_dl_workers=0,
            kl_weight=1e-6,
        )

        trainer.train()
        results = trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.1)
        self.assertLess(results['MSE'], 0.01)


class TestTrainingSinesDataset(unittest.TestCase):
    def generate_4d_grid(self, N, frequencies):
        """
        Generate a 4D grid of f(x, y, z) values with spatial frequencies.

        Args:
            N (int): Number of grid points along each axis (grid will be size^3).
            frequencies (list of tuples): List of spatial frequencies (fx, fy, fz).

        Returns:
            grid (numpy.ndarray): 4D array of shape (size, size, size, len(frequencies)).
        """
        # Define the spatial grid
        x = np.linspace(0, 2 * np.pi, N)
        y = np.linspace(0, 2 * np.pi, N)
        z = np.linspace(0, 2 * np.pi, N)

        # Create a 3D mesh grid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Initialize the 4D grid
        grid = np.zeros((N, N, N, len(frequencies)))

        # Populate the grid with sine and cosine components
        for i, (fx, fy, fz) in enumerate(frequencies):
            grid[..., i] = np.sin(fx * X) * np.cos(fy * Y) * np.sin(fz * Z)

        self.data_low_freq = torch.tensor(grid[..., 0]).float().unsqueeze(0).unsqueeze(0)
        self.train_ds_low_freq = TestDataset(self.data_low_freq)

        self.data_med_freq = torch.tensor(grid[..., 1]).float().unsqueeze(0).unsqueeze(0)
        self.train_ds_med_freq = TestDataset(self.data_med_freq)

        self.data_high_freq = torch.tensor(grid[..., 2]).float().unsqueeze(0).unsqueeze(0)
        self.train_ds_high_freq = TestDataset(self.data_high_freq)

    def setUp(self):
        # Parameters
        grid_size = 32
        spatial_frequencies = [
            (1, 1, 1),  # Low-frequency component
            (4, 4, 4),  # Mid-frequency component
            (16, 16, 16),  # High-frequency component
        ]
        self.generate_4d_grid(grid_size, spatial_frequencies)

    def test_all_models_low_freq(self):
        for model_fn, model_name in zip(models_to_compare, model_names):
            model = model_fn(im_shape=(32, 32, 32), dim=16)
            trainer = MyVAETrainer(
                model=model,
                dataset_train=self.train_ds_low_freq,
                dataset_val=self.train_ds_low_freq,
                train_batch_size=1,
                train_lr=5e-4,
                train_num_epochs=250,
                save_and_sample_every=5000,
                results_folder=f'test_output/comparison/sines_low_freq/{model_name}',
                cpu_only=True,
                num_dl_workers=0,
                kl_weight=1e-6,
            )

            trainer.train()

    def test_all_models_med_freq(self):
        for model_fn, model_name in zip(models_to_compare, model_names):
            model = model_fn(im_shape=(32, 32, 32), dim=16)
            trainer = MyVAETrainer(
                model=model,
                dataset_train=self.train_ds_med_freq,
                dataset_val=self.train_ds_med_freq,
                train_batch_size=1,
                train_lr=5e-4,
                train_num_epochs=250,
                save_and_sample_every=5000,
                results_folder=f'test_output/comparison/sines_med_freq/{model_name}',
                cpu_only=True,
                num_dl_workers=0,
                kl_weight=1e-6,
            )

            trainer.train()

    def test_baseline_block_0_low_freq(self):
        self._test_baseline_block_0('test_output_sines_low_freq_baseline_0', self.train_ds_low_freq)

    def test_baseline_block_0_med_freq(self):
        self._test_baseline_block_0('test_output_sines_med_freq_baseline_0', self.train_ds_med_freq)

    def test_baseline_block_0_high_freq(self):
        self._test_baseline_block_0('test_output_sines_high_freq_baseline_0', self.train_ds_high_freq)

    def _test_baseline_block_0(self, outname, dataset):
        model = ConvAutoencoderBaseline(
            dim=16,
            dim_mults=(1, 2, 2, 2),
            channels=1,
            z_channels=4,
            block_type=0,
            im_shape=(32, 32, 32)
        )

        trainer = MyAETrainer(
            model=model,
            dataset_train=dataset,
            dataset_val=dataset,
            train_batch_size=1,
            train_lr=5e-4,
            train_num_epochs=150,
            save_and_sample_every=50,
            results_folder=outname,
            cpu_only=True,
            num_dl_workers=0,
        )

        trainer.train()
        results = trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.1)
        self.assertLess(results['MSE'], 0.01)

    def test_vae_block_1_low_freq(self):
        self._test_vae_block_1('test_output_sines_low_freq_vae_1', self.train_ds_low_freq)

    def test_vae_block_1_med_freq(self):
        self._test_vae_block_1('test_output_sines_med_freq_vae_1', self.train_ds_med_freq)

    def test_vae_block_1_high_freq(self):
        self._test_vae_block_1('test_output_sines_high_freq_vae_1', self.train_ds_high_freq)

    def _test_vae_block_1(self, outname, dataset):
        model = VariationalAutoEncoder3D(
            dim=16,
            dim_mults=(1, 2, 2, 2),
            channels=1,
            z_channels=4,
            block_type=1,
            im_shape=(32, 32, 32),
            group_norm_size=2,
            final_kernel_size=1,
        )
        trainer = MyVAETrainer(
            model=model,
            dataset_train=dataset,
            dataset_val=dataset,
            train_batch_size=1,
            train_lr=5e-4,
            train_num_epochs=150,
            save_and_sample_every=50,
            results_folder=outname,
            cpu_only=True,
            num_dl_workers=0,
            kl_weight=1e-6,
            sample_posterior=False,
        )
        trainer.train()
        results = trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.1)
        self.assertLess(results['MSE'], 0.01)

    def test_equivariant_norm_variant_1_low_freq(self):
        self._test_vae_block_1('test_output_sines_low_freq_evae_1', self.train_ds_low_freq)

    def test_equivariant_norm_variant_1_med_freq(self):
        self._test_vae_block_1('test_output_sines_med_freq_evae_1', self.train_ds_med_freq)

    def test_equivariant_norm_variant_1_high_freq(self):
        self._test_vae_block_1('test_output_sines_high_freq_evae_1', self.train_ds_high_freq)

    def _test_equivariant_vae_norm_variant_1(self, outname, dataset):
        model = EquivariantVariationalAutoEncoder3D(
            dim=16,
            dim_mults=(1, 2, 2),
            channels=1,
            z_channels=32,
            im_shape=(32, 32, 32),
            use_norm=False,
            act_type='silu',
        )

        trainer = MyVAETrainer(
            model=model,
            dataset_train=dataset,
            dataset_val=dataset,
            train_batch_size=1,
            train_lr=1e-4,
            train_num_epochs=200,
            save_and_sample_every=50,
            results_folder=outname,
            cpu_only=True,
            num_dl_workers=0,
            kl_weight=1e-6,
            sample_posterior=False,
        )

        trainer.train()
        results = trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.1)
        self.assertLess(results['MSE'], 0.01)


    def test_equivariant_norm_variant_2_low_freq(self):
        self._test_vae_block_1('test_output_sines_low_freq_evae_2', self.train_ds_low_freq)

    def test_equivariant_norm_variant_2_med_freq(self):
        self._test_vae_block_1('test_output_sines_med_freq_evae_2', self.train_ds_med_freq)

    def test_equivariant_norm_variant_2_high_freq(self):
        self._test_vae_block_1('test_output_sines_high_freq_evae_2', self.train_ds_high_freq)

    def _test_equivariant_vae_norm_variant_2(self, outname, dataset):
        model = EquivariantVariationalAutoEncoder3D(
            dim=16,
            dim_mults=(1, 2, 2),
            channels=1,
            z_channels=32,
            im_shape=(32, 32, 32),
            use_norm=False,
            act_type='relu',
        )

        trainer = MyVAETrainer(
            model=model,
            dataset_train=dataset,
            dataset_val=dataset,
            train_batch_size=1,
            train_lr=1e-4,
            train_num_epochs=200,
            save_and_sample_every=50,
            results_folder=outname,
            cpu_only=True,
            num_dl_workers=0,
            kl_weight=1e-6,
            sample_posterior=False,
        )

        trainer.train()
        results = trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.1)
        self.assertLess(results['MSE'], 0.01)



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

    def test_all_models(self):
        for model_fn, model_name in zip(models_to_compare, model_names):
            model = model_fn(im_shape=(16, 16, 16), dim=24)
            trainer = MyVAETrainer(
                model=model,
                dataset_train=self.train_ds,
                dataset_val=self.train_ds,
                train_batch_size=self.b,
                train_lr=1e-4,
                train_num_epochs=500,
                save_and_sample_every=5000,
                results_folder=f'test_output/comparison/squares/{model_name}',
                cpu_only=True,
                num_dl_workers=0,
                kl_weight=1e-6,
            )

            trainer.train()

            break # TODO

    def test_baseline_block_0(self):
        model = ConvAutoencoderBaseline(
            dim=24,
            dim_mults=(1, 2, 2, 2),
            channels=1,
            z_channels=4,
            block_type=0,
            im_shape=(16, 16, 16)
        )

        trainer = MyVAETrainer(
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

        trainer = MyVAETrainer(
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

        trainer = MyVAETrainer(
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

    def test_equivariant_vae_no_norm_silu(self):
        model = EquivariantVariationalAutoEncoder3D(
            dim=24,
            dim_mults=(1, 2),
            channels=1,
            z_channels=32,
            im_shape=(16, 16, 16),
            use_norm=False,
            act_type='silu'
        )

        trainer = MyVAETrainer(
            model=model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=self.b,
            train_lr=1e-3,
            train_num_epochs=200,
            save_and_sample_every=50,
            results_folder='test_output_squares_equivariant_vae',
            cpu_only=True,
            num_dl_workers=0,
            kl_weight=1e-6,
            sample_posterior=False,
        )

        trainer.train()
        results = trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.1)
        self.assertLess(results['MSE'], 0.01)

    def test_equivariant_vae_no_norm_relu(self):
        model = EquivariantVariationalAutoEncoder3D(
            dim=24,
            dim_mults=(1, 2),
            channels=1,
            z_channels=32,
            im_shape=(16, 16, 16),
            use_norm=False,
            act_type='relu'
        )

        trainer = MyVAETrainer(
            model=model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=self.b,
            train_lr=1e-3,
            train_num_epochs=200,
            save_and_sample_every=50,
            results_folder='test_output_squares_equivariant_vae',
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

        trainer = MyVAETrainer(
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

    def test_all_models(self):
        for model_fn, model_name in zip(models_to_compare, model_names):
            model = model_fn(im_shape=(32, 32, 32), dim=32)
            trainer = MyVAETrainer(
                model=model,
                dataset_train=self.train_ds,
                dataset_val=self.train_ds,
                train_batch_size=1,
                train_lr=5e-4,
                train_num_epochs=500,
                save_and_sample_every=1000,
                results_folder=f'test_output/comparison/realdata/{model_name}',
                cpu_only=True,
                num_dl_workers=0,
                kl_weight=1e-6,
            )

            trainer.train()

    def test_baseline_block_0(self):
        model = ConvAutoencoderBaseline(
            dim=32,
            dim_mults=(1, 2, 2),
            channels=1,
            z_channels=1,
            block_type=0,
        )

        trainer = MyVAETrainer(
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

        trainer = MyVAETrainer(
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




if __name__ == '__main__':
    unittest.main()