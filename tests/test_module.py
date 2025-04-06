import unittest

import numpy as np

import autograd


class DummyAdd(autograd.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, x):
        return x + self.constant

class DummyMul(autograd.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, x):
        return x * self.constant

class TestModules(unittest.TestCase):
    def test_sequential(self):
        seq = autograd.Sequential(DummyMul(2), DummyAdd(3))
        x = autograd.Tensor(np.array([1, 2, 3], dtype=np.float32), requires_grad=True)
        out = seq(x)
        expected = np.array([1 * 2 + 3, 2 * 2 + 3, 3 * 2 + 3], dtype=np.float32)
        np.testing.assert_array_equal(out.data, expected, "Sequential forward pass did not produce expected output.")

    def test_dense(self):
        dense = autograd.Dense(4, 3)
        params = list(dense.parameters())
        self.assertEqual(len(params), 2, "Invalid number of parameters in Dense module.")
        x = autograd.Tensor(np.ones((2, 4), dtype=np.float32), requires_grad=True)
        out = dense(x)
        self.assertEqual(out.data.shape, (2, 3), "Dense forward pass output shape is incorrect.")

    def test_conv2d(self):
        b, h, w, c = 1, 5, 5, 3
        cout = 2
        kernel_size = (3, 3)
        conv = autograd.Conv2D(c, cout, kernel_size, 1)
        x = autograd.Tensor(np.ones((b, h, w, c), dtype=np.float32), requires_grad=True)
        kh, kw = kernel_size
        expected_h = h - kh + 1
        expected_w = w - kw + 1
        y = conv(x)
        expected_shape = (b, expected_h, expected_w, cout)
        self.assertEqual(y.data.shape, expected_shape, "Conv2D forward pass output shape is incorrect.")

    def test_avgpool2d(self):
        kernel_size = (2, 2)
        avgpool = autograd.AvgPool2D(kernel_size)
        x = autograd.Tensor(np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32), requires_grad=True)
        out_tensor = avgpool(x)
        expected_shape = (1, 1, 1, 2)
        self.assertEqual(out_tensor.data.shape, expected_shape, "AvgPool2D forward pass output shape is incorrect.")
