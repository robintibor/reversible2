import torch as th
from torch import nn
from reversible.revnet import ViewAs
from conv_spectral_norm import conv_spectral_norm

def inv_sigmoid(p, eps=1e-8):
    return th.log(p + eps) - th.log(1 - p + eps)


class DSFTransform(nn.Module):
    def __init__(self, n_features, n_units):
        super(DSFTransform, self).__init__()
        self.weights = nn.Parameter(th.randn(n_features, n_units) * 0.01,
                                    requires_grad=True)
        self.bias = nn.Parameter(th.randn(n_features, n_units) * 0.01,
                                 requires_grad=True)
        # make close to saturation of sigmoid to make transformation close
        # to identity at start
        self.alphas = nn.Parameter(th.randn(n_features, n_units) * 0.01 + 3,
                                   requires_grad=True)

    def forward(self, x):
        return inv_sigmoid(
            th.mean(th.sigmoid(self.alphas.unsqueeze(0)) * th.sigmoid(
                x.unsqueeze(2) * th.exp(self.weights).unsqueeze(0)
                + self.bias.unsqueeze(0)),
                    dim=2))


"""
class DSFTransform(nn.Module):
    def __init__(self, n_units):
        super(DSFTransform, self).__init__()
        self.weights = nn.Parameter(th.randn(200) * 0.01,
                                    requires_grad=True)
        self.bias = nn.Parameter(th.randn(200) * 0.01,
                                 requires_grad=True)
        # make close to saturation of sigmoid to make transformation close
        # to identity at start
        self.alphas = nn.Parameter(th.randn(200) * 0.01 + 4,
                                   requires_grad=True)

    def forward(self, x):
        return inv_sigmoid(
            th.mean(th.sigmoid(self.alphas.unsqueeze(0)) * th.sigmoid(
                x * th.exp(self.weights).unsqueeze(0) + self.bias.unsqueeze(0)),
                    dim=1))"""


class DistTransformResNet(nn.Module):
    def __init__(self, n_steps, n_features, n_units, ):
        super(DistTransformResNet, self).__init__()
        self.subnets = nn.Sequential()
        for i_step in range(n_steps):
            subnet = nn.Sequential(
                ViewAs((-1, n_features), (-1, n_features, 1, 1)),
                conv_spectral_norm(
                    nn.Conv2d(n_features, n_features * n_units, (1, 1),
                              groups=n_features),
                    1, 1, to_norm=0.9),
                nn.ReLU(),
                conv_spectral_norm(
                    nn.Conv2d(n_features * n_units, n_features, (1, 1),
                              groups=n_features),
                    1, 1, to_norm=0.9),
                ViewAs((-1, n_features, 1, 1), (-1, n_features)))
            self.subnets.add_module('sub_{:d}'.format(i_step), subnet)

    def forward(self, x):
        for subnet in self.subnets:
            x = x + subnet(x)
        return x

    def invert(self, out, fixed_point_steps=10):
        for subnet in self.subnets[::-1]:
            # x = y - f(x)
            with th.no_grad():
                x_guess = out.detach()
                for _ in range(fixed_point_steps):
                    x_guess = out - subnet(x_guess)
            # get some gradient
            x_guess = out - subnet(x_guess)
            out = x_guess
        return out


class TwoSort(nn.Module):
    def forward(self, x):
        c =  x.view(x.shape[0],x.shape[1] // 2,2)
        d = th.sort(c, dim=-1)[0]
        e = d.view(x.shape)
        return e

class DistTransformGroupSort(nn.Module):
    def __init__(self, n_steps, n_features, n_units, ):
        super(DistTransformGroupSort, self).__init__()
        self.subnets = nn.Sequential()
        for i_step in range(n_steps):
            subnet = nn.Sequential(
                ViewAs((-1, n_features), (-1, n_features, 1, 1)),
                conv_spectral_norm(
                    nn.Conv2d(n_features, n_features * n_units, (1, 1),
                              groups=n_features),
                    1, 1, to_norm=0.9),
                TwoSort(),
                conv_spectral_norm(
                    nn.Conv2d(n_features * n_units, n_features, (1, 1),
                              groups=n_features),
                    1, 1, to_norm=0.9),
                ViewAs((-1, n_features, 1, 1), (-1, n_features)))
            self.subnets.add_module('sub_{:d}'.format(i_step), subnet)

    def forward(self, x):
        for subnet in self.subnets:
            x = x + subnet(x)
        return x

    def invert(self, out, fixed_point_steps=10):
        for subnet in self.subnets[::-1]:
            # x = y - f(x)
            with th.no_grad():
                x_guess = out.detach()
                for _ in range(fixed_point_steps):
                    x_guess = out - subnet(x_guess)
            # get some gradient
            x_guess = out - subnet(x_guess)
            out = x_guess
        return out

