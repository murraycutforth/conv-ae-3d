import unittest
import logging
import torch
import os
from conv_ae_3d.trainer import MyAETrainer
from conv_ae_3d.models.baseline_model import ConvAutoencoderBaseline


logging.basicConfig(level=logging.INFO)


class TestCheckpointing(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data, use the same seed for consistency each time
        torch.manual_seed(0)
        self.train_ds = torch.randn(1, 1, 32, 32, 32)

    def test_save_checkpoints(self):
        model = ConvAutoencoderBaseline(
            image_shape=(32, 32, 32),
            feat_map_sizes=(16, 32, 64)
        )

        trainer = MyAETrainer(
            model=model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=2,
            train_lr=1e-3,
            train_num_epochs=10,  # Train for a few more epochs
            save_and_sample_every=5,
            results_folder='test_output',
            cpu_only=True,
            num_dl_workers=0,
        )

        trainer.train()

        results = trainer.mean_val_metrics

        self.assertLess(results['MAE'], 0.7)
        self.assertLess(results['MSE'], 0.7)

        self.assertTrue(os.path.exists(os.path.join('test_output', 'model-5.pt')))
        self.assertTrue(os.path.exists(os.path.join('test_output', 'model-10.pt')))

    def test_load_mismatched_model(self):

        self.test_save_checkpoints()

        model = ConvAutoencoderBaseline(
            image_shape=(32, 32, 32),
            feat_map_sizes=(16, 32, 32)
        )

        with self.assertRaises(RuntimeError):
            _ = MyAETrainer(
                model=model,
                dataset_train=self.train_ds,
                dataset_val=self.train_ds,
                train_batch_size=2,
                train_lr=1e-3,
                train_num_epochs=10,  # Train for a few more epochs
                save_and_sample_every=10,
                results_folder='test_output',
                cpu_only=True,
                num_dl_workers=0,
                restart_from_milestone=10,
                restart_dir='test_output'
            )

    def test_uninitialised_model(self):
        model = ConvAutoencoderBaseline(
            image_shape=(32, 32, 32),
            feat_map_sizes=(16, 32, 64)
        )

        trainer = MyAETrainer(
            model=model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=2,
            train_lr=1e-3,
            train_num_epochs=10,  # Train for a few more epochs
            save_and_sample_every=10,
            results_folder='test_output',
            cpu_only=True,
            num_dl_workers=0,
        )

        trainer.evaluate_metrics()
        results = trainer.mean_val_metrics

        self.assertGreater(results['MAE'], 0.8)
        self.assertGreater(results['MSE'], 1.2)

    def test_train_and_reload_forward(self):
        model = ConvAutoencoderBaseline(
            image_shape=(32, 32, 32),
            feat_map_sizes=(16, 32, 64)
        )

        trainer = MyAETrainer(
            model=model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=2,
            train_lr=1e-3,
            train_num_epochs=3,  # Train for a few more epochs
            save_and_sample_every=3,
            results_folder='test_output',
            cpu_only=True,
            num_dl_workers=0,
            restart_from_milestone=None,
        )

        trainer.train()

        output = trainer.model(next(iter(trainer.dl_val)))

        model = ConvAutoencoderBaseline(
            image_shape=(32, 32, 32),
            feat_map_sizes=(16, 32, 64)
        )

        reloaded_trainer = MyAETrainer(
            model=model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=2,
            train_lr=1e-4,
            train_num_epochs=3,  # Train for a few more epochs
            save_and_sample_every=3,
            results_folder='test_output',
            cpu_only=True,
            num_dl_workers=0,
            restart_from_milestone=3,
            restart_dir='test_output'
        )

        reloaded_output = reloaded_trainer.model(next(iter(reloaded_trainer.dl_val)))

        self.assertTrue(torch.allclose(output, reloaded_output))

    def test_load_checkpoint_and_evaluate(self):
        self.test_save_checkpoints()

        model = ConvAutoencoderBaseline(
            image_shape=(32, 32, 32),
            feat_map_sizes=(16, 32, 64)
        )

        trainer = MyAETrainer(
            model=model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=2,
            train_lr=1e-3,
            train_num_epochs=10,  # Train for a few more epochs
            save_and_sample_every=10,
            results_folder='test_output',
            cpu_only=True,
            num_dl_workers=0,
            restart_from_milestone=10,
            restart_dir='test_output'
        )

        trainer.evaluate_metrics()
        results = trainer.mean_val_metrics

        self.assertLess(results['MAE'], 0.6)
        self.assertLess(results['MSE'], 0.6)

    def test_load_checkpoint_and_continue_training(self):
        self.test_save_checkpoints()

        model = ConvAutoencoderBaseline(
            image_shape=(32, 32, 32),
            feat_map_sizes=(16, 32, 64)
        )

        trainer = MyAETrainer(
            model=model,
            dataset_train=self.train_ds,
            dataset_val=self.train_ds,
            train_batch_size=2,
            train_lr=1e-3,
            train_num_epochs=10,  # Train for a few more epochs
            save_and_sample_every=10,
            results_folder='test_output',
            cpu_only=True,
            num_dl_workers=0,
            restart_from_milestone=10,
            restart_dir='test_output'
        )

        trainer.train()

        results = trainer.mean_val_metrics

        self.assertTrue(os.path.exists(os.path.join('test_output', 'model-20.pt')))
        self.assertEqual(trainer.epoch, 20)
        self.assertLess(results['MAE'], 0.5)
        self.assertLess(results['MSE'], 0.4)


if __name__ == '__main__':
    unittest.main()