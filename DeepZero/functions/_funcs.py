from .DeepZeroFunction import DeepZeroFunction
from DeepZero import DeepZeroVar
import numpy as np


class Square(DeepZeroFunction):

    def forward(self, x) -> DeepZeroVar:
        return DeepZeroVar(np.power(x.data, 2))

    def backward(self, gy):
        x = self.last_inputs.data
        gx = 2 * x * gy
        return gx


class Exp(DeepZeroFunction):

    def forward(self, x) -> DeepZeroVar:
        return DeepZeroVar(np.exp(x.data))

    def backward(self, gy):
        x = self.last_inputs.data
        gx = np.exp(x) * gy
        return gx


__all__ = [
    "Square",
    "Exp"
]
