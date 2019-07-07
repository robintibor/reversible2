import torch as th
from torch import nn

class ZeroPadChans(nn.Module):
    def __init__(self, n_per_side):
        super(ZeroPadChans, self).__init__()
        self.n_per_side = n_per_side

    def forward(self, x):
        if self.n_per_side > 0:
            return th.cat((th.zeros_like(x[:, :self.n_per_side]),
                           x,
                           th.zeros_like(x[:, :self.n_per_side]),
                           ), dim=1)
        else:
            return x

    def invert(self, y):
        if self.n_per_side > 0:
            return y[:, self.n_per_side:-self.n_per_side]
        else:
            return y