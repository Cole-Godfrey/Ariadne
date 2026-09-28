"""NumPy implementations of SGD and Adam."""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class SGD:
    def __init__(self, parameters: list[Tensor], lr: float):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    def step(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.data -= self.lr * parameter.grad


class Adam(SGD):
    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.steps = 0
        self.m = [np.zeros_like(parameter.data) for parameter in parameters]
        self.v = [np.zeros_like(parameter.data) for parameter in parameters]

    def step(self) -> None:
        self.steps += 1
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            self.m[index] = self.beta1 * self.m[index] + (1 - self.beta1) * parameter.grad
            self.v[index] = self.beta2 * self.v[index] + (1 - self.beta2) * parameter.grad**2
            corrected_m = self.m[index] / (1 - self.beta1**self.steps)
            corrected_v = self.v[index] / (1 - self.beta2**self.steps)
            parameter.data -= self.lr * corrected_m / (np.sqrt(corrected_v) + self.eps)
