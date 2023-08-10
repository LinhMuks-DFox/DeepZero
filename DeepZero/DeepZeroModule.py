import abc


class DeepModule(abc.ABC):

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def backward(self, x):
        raise NotImplementedError
