from . import _type_def as td
from .DeepZeroModule import DeepModule


class ModuleSequence(DeepModule):

    def __init__(self, *args):
        super(ModuleSequence, self).__init__()
        self.modules_: td.ModuleList = [*args]

    def forward(self, x):
        ret = x
        for module in self.modules_:
            ret = module.forward(ret)
        return ret

    def backward(self, x):
        pass
