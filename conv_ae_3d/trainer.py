import json
import typing
from pathlib import Path
import logging

import accelerate
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from conv_ae_3d.metrics import MetricType, compute_metrics_single_array

logger = logging.getLogger(__name__)


class MyAETrainer():
    """High level class for training a general 3D autoencoder. Provides interface for training, saving, and loading
    models. Also includes plotting functions.
    """
    def __init__(
            self,
            model: nn.Module,
            dataset_train: Dataset,
            dataset_val: Dataset,
            train_batch_size = 32,
            train_lr = 1e-4,
            train_num_epochs = 100,
            save_and_sample_every = 10,
            adam_betas = (0.9, 0.99),
            l2_reg: float = 1e-4,
            results_folder: typing.Optional[str] = './results',
            amp = False,
            mixed_precision_type = 'fp16',
            cpu_only = False,
            num_dl_workers = 0,
            loss: nn.Module = nn.MSELoss(),
            lr_scheduler = None,
            lr_scheduler_kwargs = None,
            restart_from_milestone: typing.Optional[int] = None,
            metric_types: typing.List[MetricType] = (MetricType.MSE, MetricType.MAE, MetricType.LINF)
    ):
        super().__init__()

        self.accelerator = Accelerator(
            dataloader_config=accelerate.DataLoaderConfiguration(split_batches=False),
            mixed_precision=mixed_precision_type if amp else 'no',
            cpu=cpu_only,
        )

        self.model = model
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.train_num_epochs = train_num_epochs
        self.epoch = 0
        self.dataset_val = dataset_val
        self.loss = loss
        self.metric_types = metric_types
        self.restart_from_milestone = restart_from_milestone

        # dataset and dataloader
        dl = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=num_dl_workers)
        dl_val = DataLoader(dataset_val, batch_size=train_batch_size, shuffle=False, pin_memory=False, num_workers=num_dl_workers)

        logger.info(f"Dataset length: {len(dataset_train)}")
        logger.info(f"Dataset val length: {len(dataset_val)}")
        logger.info(f"Number of batches in train dataloader: {len(dl)}")
        logger.info(f"Number of batches in val dataloader: {len(dl_val)}")
        logger.debug("Dataloaders constructed")

        # Check the size of dataset images
        first_batch = next(iter(dl))
        assert len(first_batch.shape) == 5, 'Expected 4D tensor for 3D convolutional model'

        self.opt = Adam(model.parameters(), lr=train_lr, betas=adam_betas, weight_decay=l2_reg)
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.opt, **lr_scheduler_kwargs)
        else:
            self.lr_scheduler = None

        self.mean_val_metric_history = []
        self.mean_train_metric_history = []

        if results_folder is None:
            logger.info(f'No results folder specified, skipping most output')
            self.results_folder = None
        else:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        if self.restart_from_milestone is not None:
            assert exists(self.results_folder)
            self.load(self.restart_from_milestone)
            self.train_num_epochs += self.epoch
            logger.info(f'Restarting training from milestone {self.restart_from_milestone}')
            logger.info(f'Current epoch: {self.epoch}')
            logger.info(f'New total number of epochs: {self.train_num_epochs}')
            logger.info(f'Current learning rate: {self.opt.param_groups[0]["lr"]}')

        self.dl, self.dl_val, self.model, self.opt = self.accelerator.prepare(dl, dl_val, model, self.opt)
        if exists(self.lr_scheduler):
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)

        logger.info(f'Model and data moved to device {self.accelerator.device}')

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        assert exists(self.results_folder), "Results folder does not exist"

        if not self.accelerator.is_local_main_process:
            return

        data = {
            'epoch': self.epoch,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if exists(self.lr_scheduler) else None,
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        logger.info(f'Saving model at epoch {self.epoch}')

    def load(self, milestone: int) -> None:
        """Load model checkpoint from disk
        """
        assert exists(self.results_folder), "Results folder does not exist"
        assert (self.results_folder / f'model-{milestone}.pt').exists(), f"Model checkpoint {milestone} does not exist"
        assert isinstance(self.model, nn.Module), "Model must be an instance of nn.Module (make sure it has not been modified through accelerator.prepare())"

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'),
                          map_location=self.accelerator.device,
                          weights_only=False)

        self.model.load_state_dict(data['model'])

        self.epoch = data['epoch']
        self.opt.load_state_dict(data['opt'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

        if exists(self.lr_scheduler) and exists(data['lr_scheduler']):
            self.lr_scheduler.load_state_dict(data['lr_scheduler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        if accelerator.is_main_process:
            logger.info(f'Accelerate parallelism strategy: {accelerator.state.distributed_type}')

        logger.info(f'[Accelerate device {device}] Training started')

        loss_history = []

        if exists(self.results_folder):
            self.plot_intermediate_val_samples()

        accelerator.wait_for_everyone()

        self.evaluate_metrics()

        with tqdm(initial = self.epoch, total = self.train_num_epochs, disable=not accelerator.is_main_process) as pbar:

            while self.epoch < self.train_num_epochs:

                epoch_loss = []

                for data in self.dl:
                    data = data.to(device)

                    with self.accelerator.autocast():
                        pred = self.model(data)
                        loss = self.loss(pred, data)

                    epoch_loss.append(float(loss.item()))

                    self.accelerator.backward(loss)

                    pbar.set_description(f'Current batch loss: {loss.item():.4f}')

                    accelerator.wait_for_everyone()

                    self.opt.step()
                    self.opt.zero_grad()

                    accelerator.wait_for_everyone()

                self.epoch += 1
                if exists(self.lr_scheduler):
                    self.lr_scheduler.step()

                epoch_loss = np.mean(epoch_loss)
                loss_history.append(epoch_loss)

                if accelerator.is_main_process:
                    logger.info(str(pbar))

                if (self.epoch % self.save_and_sample_every == 0) and self.epoch != self.train_num_epochs:
                    self.evaluate_metrics()

                    if exists(self.results_folder):
                        self.save(self.epoch)
                        self.plot_intermediate_val_samples()
                        self.write_loss_history(loss_history)

                pbar.update(1)

        if exists(self.results_folder):
            self.write_loss_history(loss_history)
            self.plot_metric_history()
            self.write_all_val_set_predictions()
            self.plot_final_val_samples()
            self.save(self.epoch)

        self.evaluate_metrics()
        logger.info(f'[Accelerate device {self.accelerator.device}] Training complete!')

    def write_loss_history(self, loss_history):
        """Write the loss history to file as a json file and a png plot
        """
        if self.accelerator.is_main_process:
            loss_history_path = self.results_folder / 'loss_history.json'
            with open(loss_history_path, 'w') as f:
                json.dump(loss_history, f)
            logger.info(f'Loss history written to {loss_history_path}')

            # Also write png loss plot
            fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
            ax.plot(loss_history)
            ax.set_yscale('log')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            fig.tight_layout()
            fig.savefig(self.results_folder / 'loss_history.png')
            plt.close(fig)

    def write_all_val_set_predictions(self):
        """In this method we make predictions on the val set, and write out all predictions to file.

        Note: the ordering is not guaranteed to be consistent if multiprocessing
        """
        outdir = self.results_folder / 'final_val_predictions'
        outdir.mkdir(exist_ok=True)

        if self.accelerator.is_main_process:
            logger.info(f'Writing all val set predictions to {outdir}')

        for i, (pred, data) in enumerate(self.run_inference(self.dl_val, max_n_batches=None)):
            if self.accelerator.is_main_process:
                np.savez_compressed(outdir / f"{i}.npz", pred=pred, data=data)

    def evaluate_metrics(self):
        df_val = self._evaluate_metrics_inner(self.dl_val, max_n_batches=len(self.dl_val))
        df_train = self._evaluate_metrics_inner(self.dl, max_n_batches=len(self.dl_val))

        self.mean_val_metric_history.append((self.epoch, df_val.mean()))
        self.mean_train_metric_history.append((self.epoch, df_train.mean()))

        logger.info(f'Mean validation metrics at epoch {self.epoch}: \n{df_val.mean()}')

        if self.accelerator.is_main_process and exists(self.results_folder):
            metric_outdir = self.results_folder / 'metrics'
            metric_outdir.mkdir(exist_ok=True)

            df_val.to_csv(metric_outdir / f'val_metrics_{self.epoch}.csv')
            df_train.to_csv(metric_outdir / f'train_metrics_{self.epoch}.csv')

            logger.info(f'Written metrics to {metric_outdir / f"val_metrics_{self.epoch}.csv"} and {metric_outdir / f"train_metrics_{self.epoch}.csv"}')

        self.accelerator.wait_for_everyone()

    def run_inference(self, dataloader: torch.utils.data.DataLoader, max_n_batches: typing.Optional[int]) -> typing.Generator:
        """Returns a generator of predictions on the given dataloader
        """
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader):

                if max_n_batches is not None:
                    if i >= max_n_batches:
                        break

                pred = self.model(data)
                all_preds, all_data = self.accelerator.gather_for_metrics((pred, data))
                all_data = all_data.cpu().numpy()
                all_preds = all_preds.cpu().numpy()

                assert all_preds.shape == all_data.shape, f"Expected shape {all_data.shape}, got {all_preds.shape}"
                batch_size = all_data.shape[0]

                for j in range(batch_size):
                    yield all_preds[j].squeeze(), all_data[j].squeeze()

    def _evaluate_metrics_inner(self, dataloader, max_n_batches):
        """Evaluate model on given dataset, computing metrics

        Assumes:
         - samples are 3D
         - data is single channel
        """
        metric_results = []

        for pred, data in self.run_inference(dataloader, max_n_batches):
            row = compute_metrics_single_array(data, pred, self.metric_types)
            metric_results.append(row)

        if self.accelerator.is_main_process:
            df = pd.DataFrame(metric_results)
            logger.info(f'Computed metrics from a total of {len(df)} samples')
            return df
        else:
            return None

    @property
    def mean_val_metrics(self):
        assert len(self.mean_val_metric_history) > 0, "No validation metrics computed yet"
        return self.mean_val_metric_history[-1][1]  # self.mean_val_metric_history[i] is a tuple of (epoch, metrics_series)

    def plot_metric_history(self):
        """Plot the metric history for both training and validation sets, and also write mean metric history to file
        """
        if self.accelerator.is_main_process:
            epochs = [x[0] for x in self.mean_val_metric_history]
            df_val = pd.DataFrame([x[1] for x in self.mean_val_metric_history])
            df_train = pd.DataFrame([x[1] for x in self.mean_train_metric_history])

            n_metrics = len(self.metric_types)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols

            fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), dpi=200)

            for i, metric in enumerate(self.metric_types):
                ax = axs.flatten()[i]
                ax.plot(epochs, df_val[metric.name], label='Validation')
                ax.plot(epochs, df_train[metric.name], label='Training')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.name)
                ax.set_yscale('log')
                ax.legend()

            fig.tight_layout()
            fig.savefig(self.results_folder / 'metric_history.png')
            plt.close(fig)

            logger.info(f'Metric history plot saved to {self.results_folder / "metric_history.png"}')

    def plot_intermediate_val_samples(self, n_batches: int = 20):
        """Plot samples from the validation set, and save them to disk
        """
        n_batches = min(n_batches, len(self.dl_val))
        outdir = self.results_folder / 'intermediate_val_samples'
        outdir.mkdir(exist_ok=True)

        for i, (pred, data) in enumerate(self.run_inference(self.dl_val, max_n_batches=n_batches)):
            if self.accelerator.is_main_process:
                outpath = outdir / f"{self.epoch}_{i}_slice.png"
                write_slice_plot(outpath, data, pred)

        if self.accelerator.is_main_process:
            logger.info(f"Saved {n_batches} intermediate samples to {outdir}")

    def plot_final_val_samples(self):
        """Plot samples from the validation set, and save them to disk
        """
        outdir = self.results_folder / 'final_val_samples'
        outdir.mkdir(exist_ok=True)

        for i, (pred, data) in enumerate(self.run_inference(self.dl_val, max_n_batches=None)):
            if self.accelerator.is_main_process:
                outpath = outdir / f"final_{i}_slice.png"
                write_slice_plot(outpath, data, pred)

        logger.info(f"Saved all final val samples to {outdir}")


def exists(x):
    return x is not None


def write_slice_plot(outpath: Path, data: np.ndarray, pred: np.ndarray):
    """Write a plot of orthogonal slices through data and pred volumes
    """
    assert data.shape == pred.shape, f"Expected shape {data.shape}, got {pred.shape}"
    assert len(data.shape) == 3, "Expected 3D data"

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), dpi=200)

    for j in range(3):
        ax = axs[0, j]
        ax.set_title(f'Original ({j}-slice)')
        image = np.take(data, indices=data.shape[j] // 2, axis=j)
        im = ax.imshow(image, cmap="gray")
        fig.colorbar(im, ax=ax)

        ax = axs[1, j]
        ax.set_title(f'Reconstructed ({j}-slice)')
        image = np.take(pred, indices=pred.shape[j] // 2, axis=j)
        im = ax.imshow(image, cmap="gray")
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

    logger.info(f"Saved samples spatial slice plot to {outpath}")

    ## Create isosurface plots
    #isosurface_folder = Path(results_folder) / "isosurface_plots"
    #isosurface_folder.mkdir(exist_ok=True)
    #for i in range(n_samples):
    #    write_isosurface_plot_from_arr(all_data_reconstructed[i],
    #                                   outname=isosurface_folder / f"{name}_{i}_reconstructed.png",
    #                                   level=0.5,
    #                                   verbose=True)

    #    write_isosurface_plot_from_arr(all_data[i],
    #                                   outname=isosurface_folder / f"{name}_{i}_original.png",
    #                                   level=0.5,
    #                                   verbose=True)
