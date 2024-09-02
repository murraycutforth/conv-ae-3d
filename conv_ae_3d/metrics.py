import logging
import enum

import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import directed_hausdorff
import pandas as pd

logger = logging.getLogger(__name__)


class MetricType(enum.Enum):
    MAE = 'mae'
    MSE = 'mse'
    LINF = 'linf'
    SSIM = 'ssim'
    DICE = 'dice'
    HAUSDORFF = 'hausdorff'



def compute_mae(gt_patch, pred_patch):
    return np.linalg.norm(gt_patch.flatten() - pred_patch.flatten(), ord=1) / gt_patch.size


def compute_mse(gt_patch, pred_patch):
    return (np.linalg.norm(gt_patch.flatten() - pred_patch.flatten(), ord=2))**2 / gt_patch.size


def linf_error(gt_patch, pred_patch):
    return np.linalg.norm(gt_patch.flatten() - pred_patch.flatten(), ord=np.inf)


def ssim_error(gt_patch, pred_patch):
    return ssim(gt_patch, pred_patch, data_range=gt_patch.max() - gt_patch.min())


def dice_coefficient(gt_patch, pred_patch, level: float = 0.5):
    """Returns the dice coefficient of foreground region, obtained by thresholding the images at level
    """
    gt_patch = gt_patch > level
    pred_patch = pred_patch > level
    intersection = np.sum(gt_patch * pred_patch)
    union = np.sum(gt_patch) + np.sum(pred_patch)
    return 2 * intersection / union


def hausdorff_distance(gt_patch, pred_patch, level: float = 0.5, max_num_points: int = 100_000):
    """Returns the Hausdorff distance of the foreground region, obtained by thresholding the images at level

    Note:
        The distance is in units of voxels, assumes isotropic voxels

    Args:
        gt_patch: Ground truth patch
        pred_patch: Predicted patch
        level: Threshold level
        max_num_points: Maximum number of points to use in the distance calculation (for speed purposes)
    """
    gt_patch = gt_patch > level
    pred_patch = pred_patch > level

    gt_indices = np.argwhere(gt_patch)
    pred_indices = np.argwhere(pred_patch)

    if len(gt_indices) == 0 or len(pred_indices) == 0:
        return np.nan

    while len(gt_indices) > max_num_points:
        gt_indices = gt_indices[::2]
    while len(pred_indices) > max_num_points:
        pred_indices = pred_indices[::2]

    h_1 = directed_hausdorff(gt_indices, pred_indices)[0]
    h_2 = directed_hausdorff(pred_indices, gt_indices)[0]
    return max(h_1, h_2)


metric_type_to_function = {
    MetricType.MAE: compute_mae,
    MetricType.MSE: compute_mse,
    MetricType.LINF: linf_error,
    MetricType.SSIM: ssim_error,
    MetricType.DICE: dice_coefficient,
    MetricType.HAUSDORFF: hausdorff_distance
}


def compute_metrics_single_array(input: np.ndarray, output: np.ndarray, metric_types: list[MetricType]) -> dict:
    assert output.shape == input.shape
    assert len(output.shape) == 3

    return {
        metric_type.name: metric_type_to_function[metric_type](input, output)
        for metric_type in metric_types
    }
