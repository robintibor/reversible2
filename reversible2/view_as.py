import torch as th

class ViewAs(th.nn.Module):
    def __init__(self, dims_before, dims_after):
        super(ViewAs, self).__init__()
        self.dims_before = dims_before
        self.dims_after = dims_after

    def forward(self, x):
        for i_dim in range(len(x.size())):
            expected = self.dims_before[i_dim]
            if expected != -1:
                assert x.size()[i_dim] == expected
        return x.view(self.dims_after)

    def invert(self, features):
        for i_dim in range(len(features.size())):
            expected = self.dims_after[i_dim]
            if expected != -1:
                assert features.size()[i_dim] == expected
        features = features.view(self.dims_before)
        return features