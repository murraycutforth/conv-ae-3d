from pathlib import Path

import torch
from torch import nn

from conv_ae_3d.models.baseline_model import ConvAutoencoderBaseline
from conv_ae_3d.models.vae_model import VariationalAutoEncoder3D
from conv_ae_3d.utils import logger, DiagonalGaussianDistribution


def vae_inference_single_image(model, dataset, idx, sample_posterior=False, device=None):
    """Run inference on a single image from the dataset
    """
    model.eval()
    with torch.no_grad():
        x = dataset[idx].unsqueeze(0)

        if device is not None:
            x = x.to(device)

        posterior = model.encode(x)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        x_recon = model.decode(z)

        # Un-normalise x and x_recon
        x = x.cpu().numpy().squeeze()
        x_recon = x_recon.cpu().numpy().squeeze()

        assert x.shape == x_recon.shape, f"Shapes of x and x_recon do not match: {x.shape} vs {x_recon.shape}"
        assert x.ndim == 3, f"Expected 3D image, got {x.ndim}D image"

        x = dataset.unnormalise_array(x)
        x_recon = dataset.unnormalise_array(x_recon)

        if device is not None:
            posterior = DiagonalGaussianDistribution(posterior.parameters.cpu())

        return x, x_recon, posterior


def construct_and_load_pretrained_baseline_model(restart_dir: str,
                                                 milestone,
                                                 model_type,
                                                 dim,
                                                 dim_mults,
                                                 channels,
                                                 z_channels,
                                                 block_type,):
    """Construct the model and load pretrained weights, saved by the trainer class
    """
    if model_type == 'ae':
        model = ConvAutoencoderBaseline(
            dim=dim,
            dim_mults=dim_mults,
            channels=channels,
            z_channels=z_channels,
            block_type=block_type,
        )
    elif model_type == 'vae':
        model = VariationalAutoEncoder3D(
            dim=dim,
            dim_mults=dim_mults,
            channels=channels,
            z_channels=z_channels,
            block_type=block_type,
        )
    else:
        raise ValueError(f'Model type {model_type} not supported')

    restart_dir = Path(restart_dir)
    restart_path = restart_dir / f'model-{milestone}.pt'
    assert restart_path.exists(), f"Model checkpoint at {restart_path} does not exist"
    assert isinstance(model,
                      nn.Module), "Model must be an instance of nn.Module (make sure it has not been modified through accelerator.prepare())"

    data = torch.load(str(restart_path),
                      map_location=torch.device('cpu'))  # To be moved to GPU later during accelerate.prepare()

    model.load_state_dict(data['model'])

    logger.info(f'Loaded pretrained ConvAutoencoderBaseline model from {restart_path}')

    return model
