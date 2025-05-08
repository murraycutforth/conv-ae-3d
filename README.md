# ConvAE3D

A PyTorch implementation of 3D Convolutional Autoencoders and Variational Autoencoders for 3D data analysis.

## Overview

This project provides implementations of:
- Baseline 3D Convolutional Autoencoder
- 3D Variational Autoencoder (VAE)
- 3D Convolutional Autoencoder with Fully Connected layers
- Sliding window inference for large 3D volumes

Key features:
- Multiple block types for encoder/decoder architectures
- Support for different input dimensions and channel configurations
- Configurable training with various loss functions
- Sliding window inference with optional overlap
- Built-in metrics calculation (MAE, MSE, L-inf)

## Project Structure

- `conv_ae_3d/`: Main package directory
    - `models/`: Neural network model implementations
    - `trainer_ae.py`: Training logic for autoencoders
    - `trainer_vae.py`: Training logic for variational autoencoders
    - `sliding_window_inference.py`: Inference utilities
- `tests/`: Unit tests
- `setup.py`: Package installation configuration

## Usage

```python
from conv_ae_3d.models.vae_model import VariationalAutoEncoder3D
from conv_ae_3d.trainer_vae import MyVAETrainer

model = VariationalAutoEncoder3D(
    dim=16,
    dim_mults=(1, 2, 4, 8),
    channels=1,
    z_channels=1,
    block_type=1
)

trainer = MyVAETrainer(
    model=model,
    dataset_train=train_dataset,
    dataset_val=val_dataset,
    kl_weight=1e-6,
    sample_posterior=False
)
```