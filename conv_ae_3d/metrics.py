import logging
import enum

import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import directed_hausdorff
from conv_ae_3d.power_spectrum import compute_3d_power_spectrum

logger = logging.getLogger(__name__)


class MetricType(enum.Enum):
    MAE = 'mae'
    MSE = 'mse'
    LINF = 'linf'
    SSIM = 'ssim'
    DICE = 'dice'
    HAUSDORFF = 'hausdorff'
    REL_L1 = 'rel_L1'
    REL_L2 = 'rel_L2'
    REL_LINF = 'rel_Linf'
    REL_CONSERVATION_ERR = 'rel_conservation_err'
    PSD_ERR = 'psd_err'
    SDF_HEAVISIDE_L1 = 'sdf_heaviside_L1'
    TANH_HEAVISIDE_L1 = 'tanh_heaviside_L1'


# TODO: the code design has failed here, since we are now writing implementation-specific metrics into this library
# TODO: refactor s.t. metric function is passed in to trainer?

def compute_sdf_heaviside_l1(gt_sdf, pred_sdf):
    h_gt = np.heaviside(-gt_sdf, 0.0)
    h_pred = np.heaviside(-pred_sdf, 0.0)
    return np.linalg.norm(h_gt.flatten() - h_pred.flatten(), ord=1)


def compute_tanh_heaviside_l1(gt_tanh, pred_tanh):
    h_gt = np.heaviside(gt_tanh - 0.5, 1.0)
    h_pred = np.heaviside(pred_tanh - 0.5, 1.0)
    return np.linalg.norm(h_gt.flatten() - h_pred.flatten(), ord=1)


def compute_psd_err(gt_patch, pred_patch):
    # TODO: compute power spectrum, and then return the average relative error between the PSD at each k
    raise NotImplementedError


def compute_rel_conservation_err(gt_patch, pred_patch):
    return np.abs(np.sum(gt_patch) - np.sum(pred_patch)) / np.sum(gt_patch)


def compute_rel_l1(gt_patch, pred_patch):
    return np.linalg.norm(gt_patch.flatten() - pred_patch.flatten(), ord=1) / np.linalg.norm(gt_patch.flatten(), ord=1)


def compute_rel_l2(gt_patch, pred_patch):
    return np.linalg.norm(gt_patch.flatten() - pred_patch.flatten(), ord=2) / np.linalg.norm(gt_patch.flatten(), ord=2)


def compute_rel_linf(gt_patch, pred_patch):
    return np.linalg.norm(gt_patch.flatten() - pred_patch.flatten(), ord=np.inf) / np.linalg.norm(gt_patch.flatten(), ord=np.inf)


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
    MetricType.HAUSDORFF: hausdorff_distance,
    MetricType.REL_L1: compute_rel_l1,
    MetricType.REL_L2: compute_rel_l2,
    MetricType.REL_LINF: compute_rel_linf,
    MetricType.REL_CONSERVATION_ERR: compute_rel_conservation_err,
    MetricType.PSD_ERR: compute_psd_err,
    MetricType.SDF_HEAVISIDE_L1: compute_sdf_heaviside_l1,
    MetricType.TANH_HEAVISIDE_L1: compute_tanh_heaviside_l1,
}


def compute_metrics_single_array(input: np.ndarray, output: np.ndarray, metric_types: list[MetricType]) -> dict:
    assert output.shape == input.shape
    assert len(output.shape) == 3

    return {
        metric_type.name: metric_type_to_function[metric_type](input, output)
        for metric_type in metric_types
    }
