"""Numerical checks for the reverse-mode gradients."""

import unittest

import numpy as np

from gradforge import Adam, MLP, SGD, Tensor, cross_entropy


def finite_difference(function, array, step=1e-5):
    gradient = np.empty_like(array, dtype=np.float64)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + step
        positive = function()
        array[index] = original - step
        negative = function()
        array[index] = original
        gradient[index] = (positive - negative) / (2 * step)
    return gradient


class GradientTests(unittest.TestCase):
    def test_broadcast_and_matmul(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(3, 4))
        weight = rng.normal(size=(4, 2))
        bias = rng.normal(size=(1, 2))
        x_tensor = Tensor(x, requires_grad=True)
        weight_tensor = Tensor(weight, requires_grad=True)
        bias_tensor = Tensor(bias, requires_grad=True)
        loss = ((x_tensor @ weight_tensor + bias_tensor) * 1.7).mean()
        loss.backward()

        function = lambda: float(((x @ weight + bias) * 1.7).mean())
        np.testing.assert_allclose(x_tensor.grad, finite_difference(function, x), atol=1e-9)
        np.testing.assert_allclose(weight_tensor.grad, finite_difference(function, weight), atol=1e-9)
        np.testing.assert_allclose(bias_tensor.grad, finite_difference(function, bias), atol=1e-9)

    def test_relu_and_shared_paths(self):
        x = np.array([[-2.0, 0.7, 1.3], [1.1, -0.5, 2.0]])
        tensor = Tensor(x, requires_grad=True)
        loss = (tensor.relu() * tensor + tensor).sum()
        loss.backward()
        function = lambda: float((np.maximum(x, 0) * x + x).sum())
        np.testing.assert_allclose(tensor.grad, finite_difference(function, x), atol=1e-9)

    def test_softmax_vector_jacobian(self):
        values = np.array([[1.0, -2.0, 3.0], [0.2, 0.7, -0.4]])
        weights = np.array([[0.4, -0.2, 1.0], [-0.1, 0.3, 0.5]])
        tensor = Tensor(values, requires_grad=True)
        (tensor.softmax() * weights).sum().backward()

        def function():
            shifted = values - values.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            return float((probabilities * weights).sum())

        np.testing.assert_allclose(tensor.grad, finite_difference(function, values), atol=1e-9)

    def test_cross_entropy_is_stable_and_differentiable(self):
        values = np.array([[1000.0, 998.0, 997.0], [-1000.0, -999.0, -998.0]])
        targets = np.array([0, 2])
        tensor = Tensor(values, requires_grad=True)
        loss = cross_entropy(tensor, targets)
        self.assertTrue(np.isfinite(loss.data))
        loss.backward()
        function = lambda: float(cross_entropy(Tensor(values), targets).data)
        np.testing.assert_allclose(tensor.grad, finite_difference(function, values), atol=1e-8)

    def test_full_mlp_parameters(self):
        model = MLP((3, 4, 2), np.random.default_rng(7))
        for parameter in model.parameters():
            parameter.data = parameter.data.astype(np.float64)
        images = np.array([[0.3, 0.8, -0.2], [0.5, -0.1, 0.6]], dtype=np.float64)
        labels = np.array([0, 1])
        loss = cross_entropy(model(Tensor(images)), labels)
        loss.backward()
        for parameter in model.parameters():
            function = lambda: float(cross_entropy(model(Tensor(images)), labels).data)
            np.testing.assert_allclose(
                parameter.grad,
                finite_difference(function, parameter.data),
                atol=1e-7,
                rtol=1e-6,
            )

    def test_optimizers_update_parameters(self):
        parameter = Tensor(np.array([1.0]), requires_grad=True)
        parameter.grad = np.array([2.0])
        SGD([parameter], lr=0.1).step()
        np.testing.assert_allclose(parameter.data, [0.8])
        optimizer = Adam([parameter], lr=0.1)
        optimizer.step()
        np.testing.assert_allclose(parameter.data, [0.7], atol=1e-7)
        optimizer.zero_grad()
        self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
