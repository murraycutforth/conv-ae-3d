from pathlib import Path
import logging

import torch
from torch import nn

from conv_ae_3d.models.baseline_model import ConvAutoencoderBaseline

logger = logging.getLogger(__name__)


def construct_and_load_pretrained_baseline_model(restart_dir: str,
                                                 milestone,
                                                 image_shape,
                                                 activation,
                                                 norm,
                                                 feat_map_sizes,
                                                 linear_layer_sizes,
                                                 final_activation):
    """Construct the model and load pretrained weights, saved by the trainer class
    """
    model = ConvAutoencoderBaseline(
        image_shape=image_shape,
        activation=activation,
        norm=norm,
        feat_map_sizes=feat_map_sizes,
        linear_layer_sizes=linear_layer_sizes,
        final_activation=final_activation
    )

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