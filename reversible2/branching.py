import numpy as np
import torch as th
from torch import nn

class SplitEveryNth(nn.Module):
    def __init__(self, n_parts):
        super(SplitEveryNth, self).__init__()
        self.n_parts = n_parts

    def forward(self, x):
        xs = tuple([x[:, i::self.n_parts] for i in range(self.n_parts)])
        return xs

    def invert(self, y):
        new_y = th.zeros(
            (y[0].shape[0], y[0].shape[1] * self.n_parts,) + y[0].shape[2:],
            device=y[0].device)
        for i in range(self.n_parts):
            new_y[:, i::self.n_parts] = y[i]
        return new_y


class ChunkChans(nn.Module):
    def __init__(self, n_parts):
        super(ChunkChans, self).__init__()
        self.n_parts = n_parts

    def forward(self, x):
        xs = th.chunk(x, chunks=self.n_parts, dim=1, )
        return xs

    def invert(self, y):
        y = th.cat(y, dim=1)
        return y


class Select(nn.Module):
    def __init__(self, index):
        super(Select, self).__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]

    def invert(self, y):
        return y


class Identity(nn.Module):
    def forward(self, *x):
        return x


class CatChans(nn.Module):
    def __init__(self, n_parts=None):
        self.n_chans = None
        self.n_parts = n_parts
        super(CatChans, self).__init__()

    def forward(self, *x):
        n_chans = tuple([a_x.size()[1] for a_x in x])
        if self.n_chans is None:
            self.n_chans = n_chans
        else:
            assert n_chans == self.n_chans
        return th.cat(x, dim=1)

    def invert(self, y):
        if self.n_chans is None:
            assert self.n_parts is not None, "make forward first or supply n parts"
            xs = th.chunk(y, chunks=self.n_parts, dim=1, )
            self.n_chans = tuple([a_x.size()[1] for a_x in xs])
        else:
            xs = []
            bounds = np.insert(np.cumsum(self.n_chans), 0, 0)
            for i_b in range(len(bounds) - 1):
                xs.append(y[:, bounds[i_b]:bounds[i_b + 1]])
        return xs

class SwitchX1X2(nn.Module):
    def forward(self, x):
        x1, x2 = th.chunk(x, 2, dim=1)
        return th.cat([x2, x1], dim=1)

    def invert(self, y):
        return self.forward(y)
