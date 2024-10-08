import unittest

import torch
from conv_ae_3d.trainer import MyAETrainer
from conv_ae_3d.models.baseline_model import ConvAutoencoderBaseline
from conv_ae_3d.metrics import MetricType


class TestTrainingProcessNoise(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data, just noise
        self.train_ds = torch.randn(2, 1, 32, 32, 32)

        self.model = ConvAutoencoderBaseline(
            image_shape=(32, 32, 32),
            feat_map_sizes=(16, 32, 64)
        )

        self.trainer = MyAETrainer(
            model=self.model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=2,
            train_lr=1e-3,
            train_num_epochs=50,
            save_and_sample_every=500,
            results_folder='test_output',
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


class TestTrainingProcessSquares(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data - samples with randomly placed squares
        self.train_ds = torch.zeros(1, 1, 32, 32, 32)
        for i in range(1):
            x = torch.randint(5, 20, (1,))
            y = torch.randint(5, 20, (1,))
            z = torch.randint(5, 20, (1,))
            self.train_ds[i, 0, x:x+5, y:y+5, z:z+5] = 1

        # Initialize model
        self.model = ConvAutoencoderBaseline(
            image_shape=(32, 32, 32),
            feat_map_sizes=(16, 32, 64)
        )

        # Initialize trainer
        self.trainer = MyAETrainer(
            model=self.model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=1,
            train_lr=1e-3,
            train_num_epochs=100,
            save_and_sample_every=500,
            results_folder='test_output',
            cpu_only=True,
            num_dl_workers=0,
            metric_types=[MetricType.MAE, MetricType.MSE, MetricType.LINF, MetricType.DICE, MetricType.HAUSDORFF]
        )

    def test_training(self):
        self.trainer.train()
        results = self.trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.2)
        self.assertLess(results['MSE'], 0.05)
        self.assertLess(results['LINF'],  1)
        self.assertGreater(results['DICE'], 0.5)
        self.assertLess(results['HAUSDORFF'], 5)


class TestTrainingProcessSquareWithLinearLayers(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data - samples with randomly placed squares
        self.train_ds = torch.zeros(1, 1, 32, 32, 32)
        for i in range(1):
            x = torch.randint(5, 20, (1,))
            y = torch.randint(5, 20, (1,))
            z = torch.randint(5, 20, (1,))
            self.train_ds[i, 0, x:x+5, y:y+5, z:z+5] = 1

        # Initialize model
        self.model = ConvAutoencoderBaseline(
            image_shape=(32, 32, 32),
            feat_map_sizes=(16, 32, 64),
            linear_layer_sizes=(1000, 500),
        )

        # Initialize trainer
        self.trainer = MyAETrainer(
            model=self.model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=1,
            train_lr=1e-3,
            train_num_epochs=100,
            save_and_sample_every=500,
            results_folder='test_output',
            cpu_only=True,
            num_dl_workers=0,
            metric_types=[MetricType.MAE, MetricType.MSE, MetricType.LINF, MetricType.DICE, MetricType.HAUSDORFF]
        )

    def test_training(self):
        self.trainer.train()
        results = self.trainer.mean_val_metrics
        self.assertLess(results['MAE'], 0.2)
        self.assertLess(results['MSE'], 0.05)
        self.assertLess(results['LINF'],  1)
        self.assertGreater(results['DICE'], 0.5)
        self.assertLess(results['HAUSDORFF'], 5)

if __name__ == '__main__':
    unittest.main()