import numpy as np


class DeepZeroVar:
    def __init__(self, data: np.ndarray, requires_grad: bool = True):
        self.data = data
        self.grad = np.zeros_like(data) if requires_grad else None
        self.requires_grad = requires_grad
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)

    def backward(self):
        f = self.creator
        if f is not None:
            x = f.last_inputs
            x_grad = f.backward(self.grad)
            x.backward()


__all__ = [
    "DeepZeroVar"
]
