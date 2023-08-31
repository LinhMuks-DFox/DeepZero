from DeepZero.DeepZeroVar import DeepZeroVar


class DeepZeroFunction:

    def __call__(self, inputs: DeepZeroVar):
        data_in = inputs.data
        data_out = self.forward(data_in)
        data_out.set_creator(self)
        self.last_inputs = inputs
        self.last_outputs = data_out
        return data_out

    def forward(self, *args, **kwargs) -> DeepZeroVar:
        raise NotImplementedError()

    def backward(self, *args, **kwargs):
        raise NotImplementedError()


__all__ = [
    "DeepZeroFunction"
]