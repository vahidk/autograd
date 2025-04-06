import unittest

import numpy as np

import autograd


class TestForward(unittest.TestCase):
    def test_add(self):
        x = autograd.Tensor([1, 2, 3], requires_grad=True)  
        y = autograd.Tensor([4, 5, 6], requires_grad=True)
        z = x + y
        z.sum().backward()
        expected_z = [5, 7, 9]
        expected_dx = [1, 1, 1]
        expected_dy = [1, 1, 1]
        np.testing.assert_allclose(z.data, expected_z, err_msg="Forward addition failed")
        np.testing.assert_allclose(x.grad, expected_dx, err_msg="Gradient for x in addition is incorrect")
        np.testing.assert_allclose(y.grad, expected_dy, err_msg="Gradient for y in addition is incorrect")

    def test_add_broadcast(self):
        x = autograd.Tensor([1, 2, 3], requires_grad=True)  
        y = autograd.Tensor([4], requires_grad=True)
        z = x + y
        z.sum().backward()
        expected_z = [5, 6, 7]
        expected_dx = [1, 1, 1]
        expected_dy = [3]
        np.testing.assert_allclose(z.data, expected_z, err_msg="Forward addition failed")
        np.testing.assert_allclose(x.grad, expected_dx, err_msg="Gradient for x in addition is incorrect")
        np.testing.assert_allclose(y.grad, expected_dy, err_msg="Gradient for y in addition is incorrect")

    def test_square(self):
        x = autograd.Tensor(np.array([1, 2, 3]), requires_grad=True)
        y = x.square()
        y.sum().backward()
        expected_y = np.array([1, 4, 9])
        expected_dx = np.array([2, 4, 6])
        np.testing.assert_allclose(y.data, expected_y, err_msg="Forward square failed")
        np.testing.assert_allclose(x.grad, expected_dx, err_msg="Gradient for x in square function is incorrect")

    def test_log(self):
        x = autograd.Tensor(np.array([1, 2, 3]), requires_grad=True)
        y = x.log()
        y.sum().backward()
        expected_y = np.array([0.0, np.log(2), np.log(3)])
        expected_dx = np.array([1.0, 1/2.0, 1/3.0])
        np.testing.assert_allclose(y.data, expected_y, err_msg="Forward log failed")
        np.testing.assert_allclose(x.grad, expected_dx, err_msg="Gradient for x in log function is incorrect")

    def test_exp(self):
        x = autograd.Tensor(np.array([1, 2, 3]), requires_grad=True)
        y = x.exp()
        y.sum().backward()
        expected_y = np.array([np.exp(1), np.exp(2), np.exp(3)])
        expected_dx = np.array([np.exp(1), np.exp(2), np.exp(3)])
        np.testing.assert_allclose(y.data, expected_y, err_msg="Forward exp failed")
        np.testing.assert_allclose(x.grad, expected_dx, err_msg="Gradient for x in exp function is incorrect")

    def test_mean(self):
        x = autograd.Tensor(np.array([[1, 2, 3], [4, 5, 6]]), requires_grad=True)
        y = x.mean(axis=1)
        y.sum().backward()
        expected_y = np.array([2.0, 5.0])
        expected_dx = np.ones_like(x.data) / 3.0
        expected_dx[0, :] = 1 / 3.0
        expected_dx[1, :] = 1 / 3.0
        np.testing.assert_allclose(y.data, expected_y, err_msg="Forward mean failed")
        np.testing.assert_allclose(x.grad, expected_dx, err_msg="Gradient for x in mean function is incorrect")

    def test_max(self):
        x = autograd.Tensor(np.array([[1, 2, 3], [4, 5, 6]]), requires_grad=True)
        y = x.max(axis=1)
        y.sum().backward()
        expected_y = np.array([3, 6])
        expected_dx = np.zeros_like(x.data)
        expected_dx[0, 2] = 1
        expected_dx[1, 2] = 1
        np.testing.assert_allclose(y.data, expected_y, err_msg="Forward max failed")
        np.testing.assert_allclose(x.grad, expected_dx, err_msg="Gradient for x in max function is incorrect")


if __name__ == "__main__":
    unittest.main()
