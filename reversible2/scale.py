import torch as th
from torch import nn


class ScalingLayer(nn.Module):
    def __init__(self, shape):
        super(ScalingLayer, self).__init__()
        self.log_factors = th.nn.Parameter(
            th.zeros(shape), requires_grad=True)

    def forward(self, x):
        factors = th.exp(self.log_factors)
        return x * factors.unsqueeze(0)

    def invert(self, y):
        factors = th.exp(self.log_factors)
        return y / factors.unsqueeze(0)
