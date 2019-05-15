import torch as th


class AffineBlock(th.nn.Module):
    def __init__(self, FA, GA, FM, GM, switched_order=True, eps=0):
        super(AffineBlock, self).__init__()
        # first G before F, only to have consistent ordering of
        # parameter list compared to other code
        self.GA = GA
        self.FA = FA
        self.GM = GM
        self.FM = FM
        self.switched_order = switched_order
        self.eps = eps # is used in inverse

    def forward(self, x):
        n_chans = x.size()[1]
        assert n_chans % 2 == 0
        x1 = x[:, :n_chans // 2]
        x2 = x[:, n_chans // 2:]
        if self.switched_order:
            #y1 = self.FA(x1) + x2 * th.exp(self.FM(x1))
            #y2 = self.GA(y1) + x1 * th.exp(self.GM(y1))
            y1 = x2
            y2 = x1
        else:
            #y1 = self.FA(x2) + x1 * th.exp(self.FM(x2))
            #y2 = self.GA(y1) + x2 * th.exp(self.GM(y1))
            y1 = x1
            y2 = x2

        if self.FM is not None:
            y1 = y1 * th.exp(self.FM(y2))
        if self.FA is not None:
            y1 = y1 + self.FA(y2)
        if self.GM is not None:
            y2 = y2 * th.exp(self.GM(y1))
        if self.GA is not None:
            y2 = y2 + self.GA(y1)
        return th.cat((y1, y2), dim=1)


class AdditiveBlock(AffineBlock):
    def __init__(self, FA, GA, switched_order=True, eps=0):
        super(AdditiveBlock, self).__init__(
            FA=FA, GA=GA,FM=None, GM=None,
            switched_order=switched_order, eps=eps)


class MultiplicativeBlock(AffineBlock):
    def __init__(self, FM, GM, switched_order=True, eps=0):
        super(AdditiveBlock, self).__init__(
            FA=None, GA=None, FM=FM, GM=GM,
            switched_order=switched_order, eps=eps)
