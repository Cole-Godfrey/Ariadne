"""a small NumPy neural-network framework with reverse-mode autodiff."""

from .nn import Linear, MLP, ReLU
from .optim import Adam, SGD
from .tensor import Tensor, cross_entropy

__all__ = ["Tensor", "cross_entropy", "Linear", "ReLU", "MLP", "SGD", "Adam"]
