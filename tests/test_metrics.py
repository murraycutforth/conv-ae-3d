import unittest
import numpy as np
from src.metrics import hausdorff_distance, compute_mae, compute_mse, linf_error, ssim_error, dice_coefficient


class TestComputeMAE(unittest.TestCase):

    def test_identical_grids(self):
        gt_patch = np.ones((5, 5, 5))
        pred_patch = np.ones((5, 5, 5))
        self.assertEqual(compute_mae(gt_patch, pred_patch), 0)

    def test_single_point(self):
        gt_patch = np.zeros((5, 5, 5))
        pred_patch = np.zeros((5, 5, 5))
        gt_patch[2, 2, 2] = 1
        pred_patch[2, 2, 2] = 1
        self.assertEqual(compute_mae(gt_patch, pred_patch), 0)

    def test_different_points(self):
        gt_patch = np.zeros((5, 5, 5))
        pred_patch = np.zeros((5, 5, 5))
        gt_patch[0, 0, 0] = 1
        pred_patch[4, 4, 4] = 1
        self.assertEqual(compute_mae(gt_patch, pred_patch), 2 / 125)


class TestComputeMSE(unittest.TestCase):

    def test_identical_grids(self):
        gt_patch = np.ones((5, 5, 5))
        pred_patch = np.ones((5, 5, 5))
        self.assertEqual(compute_mse(gt_patch, pred_patch), 0)

    def test_single_point(self):
        gt_patch = np.zeros((5, 5, 5))
        pred_patch = np.zeros((5, 5, 5))
        gt_patch[2, 2, 2] = 1
        pred_patch[2, 2, 2] = 1
        self.assertEqual(compute_mse(gt_patch, pred_patch), 0)

    def test_different_points(self):
        gt_patch = np.zeros((5, 5, 5))
        pred_patch = np.zeros((5, 5, 5))
        gt_patch[0, 0, 0] = 1
        pred_patch[4, 4, 4] = 1
        self.assertAlmostEqual(compute_mse(gt_patch, pred_patch), 2 / 125)


class TestLinfError(unittest.TestCase):

    def test_identical_grids(self):
        gt_patch = np.ones((5, 5, 5))
        pred_patch = np.ones((5, 5, 5))
        self.assertEqual(linf_error(gt_patch, pred_patch), 0)

    def test_single_point(self):
        gt_patch = np.zeros((5, 5, 5))
        pred_patch = np.zeros((5, 5, 5))
        gt_patch[2, 2, 2] = 1
        pred_patch[2, 2, 2] = 1
        self.assertEqual(linf_error(gt_patch, pred_patch), 0)

    def test_different_points(self):
        gt_patch = np.zeros((5, 5, 5))
        pred_patch = np.zeros((5, 5, 5))
        gt_patch[0, 0, 0] = 1
        pred_patch[4, 4, 4] = 1
        self.assertEqual(linf_error(gt_patch, pred_patch), 1)


class TestDiceCoefficient(unittest.TestCase):

    def test_identical_grids(self):
        gt_patch = np.ones((5, 5, 5))
        pred_patch = np.ones((5, 5, 5))
        self.assertEqual(dice_coefficient(gt_patch, pred_patch), 1)

    def test_single_point(self):
        gt_patch = np.zeros((5, 5, 5))
        pred_patch = np.zeros((5, 5, 5))
        gt_patch[2, 2, 2] = 1
        pred_patch[2, 2, 2] = 1
        self.assertEqual(dice_coefficient(gt_patch, pred_patch), 1)

    def test_different_points(self):
        gt_patch = np.zeros((5, 5, 5))
        pred_patch = np.zeros((5, 5, 5))
        gt_patch[0, 0, 0] = 1
        pred_patch[4, 4, 4] = 1
        self.assertEqual(dice_coefficient(gt_patch, pred_patch), 0)




class TestHausdorffDistance(unittest.TestCase):

    def test_identical_grids(self):
        gt_patch = np.ones((5, 5, 5))
        pred_patch = np.ones((5, 5, 5))
        self.assertEqual(hausdorff_distance(gt_patch, pred_patch), 0)

    def test_single_point(self):
        gt_patch = np.zeros((5, 5, 5))
        pred_patch = np.zeros((5, 5, 5))
        gt_patch[2, 2, 2] = 1
        pred_patch[2, 2, 2] = 1
        self.assertEqual(hausdorff_distance(gt_patch, pred_patch), 0)

    def test_different_points(self):
        gt_patch = np.zeros((5, 5, 5))
        pred_patch = np.zeros((5, 5, 5))
        gt_patch[0, 0, 0] = 1
        pred_patch[4, 4, 4] = 1
        self.assertEqual(hausdorff_distance(gt_patch, pred_patch), np.sqrt(3 * (4**2)))

    def test_empty_and_non_empty(self):
        gt_patch = np.zeros((5, 5, 5))
        pred_patch = np.zeros((5, 5, 5))
        pred_patch[2, 2, 2] = 1
        self.assertIs(hausdorff_distance(gt_patch, pred_patch), np.nan)

    def test_complex_case(self):
        gt_patch = np.zeros((5, 5, 5))
        pred_patch = np.zeros((5, 5, 5))
        gt_patch[1, 1, 1] = 1
        gt_patch[3, 3, 3] = 1
        pred_patch[1, 1, 1] = 1
        pred_patch[4, 4, 4] = 1
        self.assertEqual(hausdorff_distance(gt_patch, pred_patch), np.sqrt(3 * (1**2)))


if __name__ == '__main__':
    unittest.main()