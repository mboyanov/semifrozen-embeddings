from fastai.text import RNNLearner
from torch import nn


class FrozenStrategy:

    def __init__(self):
        pass

    def fit(self, learn:RNNLearner):
        for l in learn.model.modules():
            if isinstance(l, [nn.Embedding]):
                l.weight.requires_grad = False
        for i in range(self.epochs):
            learn.fit_one_cycle(self.cycle_len, self.lr)

