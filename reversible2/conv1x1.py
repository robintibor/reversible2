# https://github.com/rosinality/glow-pytorch/blob/ddb4b65384a5f96bdfab2f07194b98c5da46ae80/model.py
from torch import nn
import torch as th
import torch.nn.functional as F

class Conv1x1(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = th.randn(in_channel, in_channel)
        q, _ = th.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        out = F.conv2d(input, self.weight)

        return out

    def invert(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class Dense1x1(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = th.randn(in_channel, in_channel)
        q, _ = th.qr(weight)
        weight = q
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        out = F.linear(input, self.weight)
        return out

    def invert(self, output):
        return F.linear(output, self.weight.inverse())
