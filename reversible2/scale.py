import torch as th
from torch import nn


def scale_to_unit_var(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    self.add.data = -th.mean(output[:, :2, 0]).data
    self.log_factor.data = th.log((1 / th.std(output[:, :2, 0], ))).data
    print("Setting log factor to", self.log_factor)
    print("Setting add to", self.add)

class ScaleAndShift(nn.Module):
    def __init__(self, ):
        super(ScaleAndShift, self).__init__()
        self.log_factor = nn.Parameter(th.zeros(1))
        self.add = nn.Parameter(th.zeros(1))

    def forward(self, x):
        return (x + self.add) * th.exp(self.log_factor)

    def invert(self, y):
        return (y / th.exp(self.log_factor)) - self.add


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

