import unittest
import torch
import numpy as np
from torch.utils.data import Dataset
from conv_ae_3d.sliding_window_inference import sliding_window_inference, sliding_window_inference_with_overlap
from conv_ae_3d.models.baseline_model import ConvAutoencoderBaseline

class Synthetic3DDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.tensor(self.data)

class TestSlidingWindowInference(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.data = torch.randn(1, 32, 32, 32).numpy()
        self.dataset = Synthetic3DDataset(self.data)
        self.model = ConvAutoencoderBaseline(
            dim=16,
            dim_mults=(1, 2),
            channels=1,
            z_channels=4,
            block_type=0,
        )
        self.device = torch.device('cpu')

    def test_sliding_window_inference(self):
        window_size = (16, 16, 16)
        output = sliding_window_inference(self.model, self.dataset[0], window_size, self.device)
        self.assertEqual(output.shape, self.data.squeeze().shape)

    def test_sliding_window_inference_with_overlap(self):
        window_size = (16, 16, 16)
        stride = (8, 8, 8)
        output = sliding_window_inference_with_overlap(self.model, self.dataset[0], window_size, stride, self.device)
        self.assertEqual(output.shape, self.data.squeeze().shape)


class OnesModel(torch.nn.Module):
    def __init__(self):
        super(OnesModel, self).__init__()

    def forward(self, x):
        return torch.ones_like(x)

class TestSlidingWindowInferenceOnesModel(unittest.TestCase):
    # This isolates the sliding window inference function from the model

    def setUp(self):
        torch.manual_seed(0)
        self.data = torch.randn(1, 32, 32, 32).numpy()
        self.dataset = Synthetic3DDataset(self.data)
        self.model = OnesModel()
        self.device = torch.device('cpu')

    def test_sliding_window_inference(self):
        window_size = (16, 16, 16)
        output = sliding_window_inference(self.model, self.dataset[0], window_size, self.device)
        self.assertEqual(output.shape, self.data.squeeze().shape)

    def test_sliding_window_inference_with_overlap(self):
        window_size = (16, 16, 16)
        stride = (8, 8, 8)
        output = sliding_window_inference_with_overlap(self.model, self.dataset[0], window_size, stride, self.device)
        self.assertEqual(output.shape, self.data.squeeze().shape)

    def test_sliding_window_inference_differences(self):
        window_size = (16, 16, 16)
        stride = (8, 8, 8)
        output_overlap_1 = sliding_window_inference_with_overlap(self.model, self.dataset[0], window_size, stride, self.device)
        stride = (16, 16, 16)
        output_overlap_2 = sliding_window_inference_with_overlap(self.model, self.dataset[0], window_size, stride, self.device)
        np.testing.assert_allclose(output_overlap_1, output_overlap_2, atol=1e-6)


if __name__ == '__main__':
    unittest.main()