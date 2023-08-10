import numpy as np
from .DeepZeroModule import DeepModule
from . import _type_def as td


class Linear(DeepModule):
    def __init__(self, feature_in: int, feature_out: int, bias: bool = True):
        self.weight_: td.NDArray = np.random.randn(feature_out, feature_in)
        self.bias_: td.NDArray = np.random.randn(feature_out)

    def forward(self, x: td.NDArray):
        return self._forward_impl(x)

    def _forward_impl(self, x: td.NDArray):
        return np.dot(self.weight_, x) + self.bias_

    def backward(self, x):
        return self._backward_impl(x)

    def _backward_impl(self, x):
        return np.dot(self.weight_, x) + self.bias_
