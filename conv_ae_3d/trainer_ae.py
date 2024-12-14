import json
import typing
from pathlib import Path
import logging

import accelerate
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from conv_ae_3d.metrics import MetricType, compute_metrics_single_array
from conv_ae_3d.trainer_base import MyTrainerBase, exists


logger = logging.getLogger(__name__)


class MyAETrainer(MyTrainerBase):
    """High level class for training a 3D autoencoder.
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
            low_data_mode: bool = False,
            adam_betas = (0.9, 0.99),
            l2_reg: float = 1e-4,
            results_folder: typing.Optional[str] = './results',
            cpu_only = False,
            num_dl_workers = 0,
            loss: nn.Module = nn.MSELoss(),
            lr_scheduler = None,
            lr_scheduler_kwargs = None,
            restart_from_milestone: typing.Optional[int] = None,
            restart_dir: typing.Optional[str] = None,
            metric_types: typing.List[MetricType] = (MetricType.MSE, MetricType.MAE, MetricType.LINF)
    ):
        super().__init__(
            model=model,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            train_batch_size=train_batch_size,
            train_lr=train_lr,
            train_num_epochs=train_num_epochs,
            save_and_sample_every=save_and_sample_every,
            low_data_mode=low_data_mode,
            adam_betas=adam_betas,
            l2_reg=l2_reg,
            results_folder=results_folder,
            cpu_only=cpu_only,
            num_dl_workers=num_dl_workers,
            loss=loss,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            restart_from_milestone=restart_from_milestone,
            restart_dir=restart_dir,
            metric_types=metric_types
        )

        # Check the size of dataset images
        first_batch = next(iter(self.dl))
        assert len(first_batch.shape) == 5, 'Expected 4D tensor for 3D convolutional model'
        z = self.model.encode(first_batch)
        self.write_run_info(first_batch, z)

    def train(self):
        """Run full training loop
        """
        accelerator = self.accelerator
        device = accelerator.device

        if accelerator.is_main_process:
            logger.info(f'Accelerate parallelism strategy: {accelerator.state.distributed_type}')

        logger.info(f'[Accelerate device {device}] Training started')

        loss_history = []
        accelerator.wait_for_everyone()
        self.evaluate_metrics()
        step = 0

        with tqdm(initial = self.epoch, total = self.train_num_epochs, disable=not accelerator.is_main_process) as pbar:

            while self.epoch < self.train_num_epochs:

                epoch_loss = []

                for data in self.dl:

                    pred = self.model(data)
                    loss = self.loss(pred, data)

                    epoch_loss.append({'Total': loss.item()})

                    self.accelerator.backward(loss)

                    accelerator.wait_for_everyone()

                    if step == 0:
                        if torch.cuda.is_available():
                            if self.accelerator.is_main_process:
                                memory_summary = torch.cuda.memory_summary()

                                logger.info(memory_summary)
                                with open(self.results_folder / 'memory_summary.txt', 'w') as f:
                                    f.write(memory_summary)

                    step += 1
                    self.opt.step()
                    self.opt.zero_grad()

                    accelerator.wait_for_everyone()

                self.epoch += 1
                if exists(self.lr_scheduler):
                    self.lr_scheduler.step()

                epoch_loss = {k: np.mean([x[k] for x in epoch_loss]) for k in epoch_loss[0]}
                loss_history.append(epoch_loss)
                pbar.set_description(f'Avg. epoch loss: Total={epoch_loss["Total"]:.3g}')

                if accelerator.is_main_process:
                    logger.info(str(pbar))

                if (self.epoch % self.save_and_sample_every == 0) and self.epoch != self.train_num_epochs:
                    self.evaluate_metrics()

                    if exists(self.results_folder):
                        #self.save(self.epoch)
                        self.plot_intermediate_val_samples()
                        self.write_loss_history(loss_history)

                pbar.update(1)

        if exists(self.results_folder):
            self.save('final')
            self.write_loss_history(loss_history)

            if not self.low_data_mode:
                self.write_all_val_set_predictions()
                self.plot_final_val_samples()

        self.evaluate_metrics()
        if exists(self.results_folder):
            self.plot_metric_history()

        logger.info(f'[Accelerate device {self.accelerator.device}] Training complete!')

