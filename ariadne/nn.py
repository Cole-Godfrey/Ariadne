"""neural-network layers built from Tensor operations."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Linear:
    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(
            rng.normal(0.0, scale, size=(in_features, out_features)).astype(np.float32),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros((1, out_features), dtype=np.float32), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def parameters(self) -> tuple[Tensor, Tensor]:
        return self.weight, self.bias


class ReLU:
    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()

    def parameters(self) -> tuple[()]:
        return ()


class MLP:
    def __init__(self, sizes: tuple[int, ...], rng: np.random.Generator):
        if len(sizes) < 2:
            raise ValueError("MLP needs at least an input and output size")
        self.layers = []
        for index, (in_features, out_features) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.layers.append(Linear(in_features, out_features, rng))
            if index < len(sizes) - 2:
                self.layers.append(ReLU())

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Tensor]:
        return [parameter for layer in self.layers for parameter in layer.parameters()]
