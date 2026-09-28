"""Scalar-loss reverse-mode autodiff over NumPy arrays."""

from __future__ import annotations

from typing import Callable

import numpy as np


def _array(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fc":
        array = array.astype(np.float32)
    return array


def _unbroadcast(gradient: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient


def _finite_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = left @ right
    if not np.isfinite(result).all():
        raise FloatingPointError("matrix multiplication produced a non-finite value")
    return result


class Tensor:
    def __init__(self, data: object, requires_grad: bool = False):
        self.data = _array(data)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._parents: tuple[Tensor, ...] = ()
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    @staticmethod
    def _as_tensor(value: Tensor | object) -> Tensor:
        return value if isinstance(value, Tensor) else Tensor(value)

    def _accumulate(self, gradient: np.ndarray) -> None:
        if self.requires_grad:
            self.grad = gradient if self.grad is None else self.grad + gradient

    def backward(self, gradient: object | None = None) -> None:
        if not self.requires_grad:
            raise ValueError("cannot backpropagate from a tensor without gradients")
        if gradient is None:
            if self.data.size != 1:
                raise ValueError("a non-scalar output needs an explicit gradient")
            gradient_array = np.ones_like(self.data)
        else:
            gradient_array = _array(gradient)
            if gradient_array.shape != self.data.shape:
                raise ValueError("gradient shape must match output shape")

        order: list[Tensor] = []
        seen: set[int] = set()

        def visit(node: Tensor) -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent in node._parents:
                visit(parent)
            order.append(node)

        visit(self)
        for node in order:
            if node._parents:
                node.grad = None
        self._accumulate(gradient_array)
        for node in reversed(order):
            node._backward()

    def __add__(self, other: Tensor | object) -> Tensor:
        other = self._as_tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad)
        out._parents = (self, other)

        def backward() -> None:
            if self.requires_grad:
                self._accumulate(_unbroadcast(out.grad, self.data.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(out.grad, other.data.shape))

        out._backward = backward
        return out

    def __radd__(self, other: Tensor | object) -> Tensor:
        return self + other

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other: Tensor | object) -> Tensor:
        return self + -self._as_tensor(other)

    def __rsub__(self, other: Tensor | object) -> Tensor:
        return self._as_tensor(other) - self

    def __mul__(self, other: Tensor | object) -> Tensor:
        other = self._as_tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad)
        out._parents = (self, other)

        def backward() -> None:
            if self.requires_grad:
                self._accumulate(_unbroadcast(out.grad * other.data, self.data.shape))
            if other.requires_grad:
                other._accumulate(_unbroadcast(out.grad * self.data, other.data.shape))

        out._backward = backward
        return out

    def __rmul__(self, other: Tensor | object) -> Tensor:
        return self * other

    def __matmul__(self, other: Tensor | object) -> Tensor:
        other = self._as_tensor(other)
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError("matmul currently supports two matrices")
        out = Tensor(_finite_matmul(self.data, other.data), self.requires_grad or other.requires_grad)
        out._parents = (self, other)

        def backward() -> None:
            if self.requires_grad:
                self._accumulate(_finite_matmul(out.grad, other.data.T))
            if other.requires_grad:
                other._accumulate(_finite_matmul(self.data.T, out.grad))

        out._backward = backward
        return out

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad)
        out._parents = (self,)

        def backward() -> None:
            if self.requires_grad:
                gradient = out.grad
                if axis is not None and not keepdims:
                    gradient = np.expand_dims(gradient, axis=axis)
                self._accumulate(np.broadcast_to(gradient, self.data.shape))

        out._backward = backward
        return out

    def mean(self) -> Tensor:
        return self.sum() * (1.0 / self.data.size)

    def relu(self) -> Tensor:
        out = Tensor(np.maximum(self.data, 0), self.requires_grad)
        out._parents = (self,)

        def backward() -> None:
            if self.requires_grad:
                self._accumulate(out.grad * (self.data > 0))

        out._backward = backward
        return out

    def softmax(self) -> Tensor:
        shifted = self.data - self.data.max(axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum(axis=-1, keepdims=True)
        out = Tensor(probabilities, self.requires_grad)
        out._parents = (self,)

        def backward() -> None:
            if self.requires_grad:
                dot = (out.grad * probabilities).sum(axis=-1, keepdims=True)
                self._accumulate(probabilities * (out.grad - dot))

        out._backward = backward
        return out


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Return mean softmax cross-entropy for integer class labels."""
    if logits.data.ndim != 2:
        raise ValueError("logits must have shape (batch, classes)")
    targets = np.asarray(targets)
    batch_size, class_count = logits.data.shape
    if targets.shape != (batch_size,) or targets.dtype.kind not in "iu":
        raise ValueError("targets must be a vector of integer class labels")
    if np.any(targets < 0) or np.any(targets >= class_count):
        raise ValueError("target outside class range")

    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    sums = exponentials.sum(axis=1, keepdims=True)
    row = np.arange(batch_size)
    loss = (np.log(sums[:, 0]) - shifted[row, targets]).mean()
    out = Tensor(loss, logits.requires_grad)
    out._parents = (logits,)

    def backward() -> None:
        if logits.requires_grad:
            probabilities = exponentials / sums
            probabilities[row, targets] -= 1.0
            logits._accumulate((out.grad / batch_size) * probabilities)

    out._backward = backward
    return out
