import torch as th
from torch import nn
from reversible.revnet import SubsampleSplitter, invert

class RFFT(nn.Module):
    def forward(self, x):
        ffted = th.rfft(x, signal_ndim=1, normalized=True)
        flattened = ffted.view(ffted.shape[0], -1)
        reduced = th.cat((flattened[:, :1], flattened[:, 2:-1]), dim=1)
        return reduced

    def invert(self, y):
        expanded = th.cat(
            (y[:, :1], th.zeros(y.shape[0], 1, device=y.device),
             y[:, 1:], th.zeros(y.shape[0], 1, device=y.device)),
            dim=1)
        unfolded = expanded.view(expanded.shape[0], expanded.shape[1] // 2, 2)
        inverted = th.irfft(unfolded, signal_ndim=1,
                            signal_sizes=((unfolded.shape[1] - 1) * 2,),
                            normalized=True)
        return inverted


class Interleave(nn.Module):
    def forward(self, x):
        assert x.shape[2] == 1
        assert x.shape[3] == 1
        dummy_splitter = SubsampleSplitter(stride=[2, 1],
                                           chunk_chans_first=False)
        interleaved = invert(dummy_splitter, x)
        interleaved = interleaved.view(*x.shape)
        return interleaved

    def invert(self, y):
        assert y.shape[2] == 1
        assert y.shape[3] == 1
        dummy_splitter = SubsampleSplitter(stride=[2, 1],
                                           chunk_chans_first=False)
        inverted = dummy_splitter(y.permute(0, 2, 1, 3))
        inverted = inverted.view(y.shape)
        return inverted
