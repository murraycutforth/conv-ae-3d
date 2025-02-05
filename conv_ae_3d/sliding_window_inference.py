import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import typing

def sliding_window_inference(
        model: nn.Module,
        data: torch.tensor,
        window_size: typing.Tuple[int, int, int],
        device: torch.device = torch.device('cpu')
) -> np.ndarray:
    """
    Perform sliding window inference on a 3D dataset using a pre-trained autoencoder model.

    Args:
        model (nn.Module): Pre-trained autoencoder model.
        window_size (tuple): Size of the sliding window (depth, height, width).
        stride (tuple): Stride of the sliding window (depth, height, width).
        device (torch.device): Device to perform inference on.

    Returns:
        np.ndarray: Reconstructed volume.
    """
    model.to(device)
    model.eval()

    data = data.squeeze().numpy()

    depth, height, width = data.shape
    d_win, h_win, w_win = window_size

    assert depth % d_win == 0, 'Depth of the volume must be divisible by the depth of the window.'
    assert height % h_win == 0, 'Height of the volume must be divisible by the height of the window.'
    assert width % w_win == 0, 'Width of the volume must be divisible by the width of the window.'

    output = np.zeros_like(data)

    with torch.no_grad():
        for d in range(0, depth - d_win + 1, d_win):
            for h in range(0, height - h_win + 1, h_win):
                for w in range(0, width - w_win + 1, w_win):
                    window = data[d:d + d_win, h:h + h_win, w:w + w_win]
                    window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    reconstructed_window = model(window_tensor).squeeze().cpu().numpy()
                    output[d:d + d_win, h:h + h_win, w:w + w_win] = reconstructed_window

    return output


def sliding_window_inference_with_overlap(
        model: nn.Module,
        data: torch.tensor,
        window_size: typing.Tuple[int, int, int],
        stride: typing.Tuple[int, int, int],
        device: torch.device = torch.device('cpu')
) -> np.ndarray:
    """
    Perform sliding window inference on a 3D dataset using a pre-trained autoencoder model with overlapping windows.

    Args:
        model (nn.Module): Pre-trained autoencoder model.
        window_size (tuple): Size of the sliding window (depth, height, width).
        stride (tuple): Stride of the sliding window (depth, height, width).
        device (torch.device): Device to perform inference on.

    Returns:
        np.ndarray: Reconstructed volume.
    """
    model.to(device)
    model.eval()

    data = data.squeeze().numpy()

    depth, height, width = data.shape
    d_win, h_win, w_win = window_size
    d_stride, h_stride, w_stride = stride

    output = np.zeros_like(data)
    count_map = np.zeros_like(data)

    with torch.no_grad():
        for d in range(0, depth - d_win + 1, d_stride):
            for h in range(0, height - h_win + 1, h_stride):
                for w in range(0, width - w_win + 1, w_stride):
                    window = data[d:d + d_win, h:h + h_win, w:w + w_win]
                    window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    reconstructed_window = model(window_tensor).squeeze().cpu().numpy()
                    output[d:d + d_win, h:h + h_win, w:w + w_win] += reconstructed_window
                    count_map[d:d + d_win, h:h + h_win, w:w + w_win] += 1

    # Average the overlapping regions
    output /= count_map

    return output
