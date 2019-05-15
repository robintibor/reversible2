from torch import nn
from reversible2.invert import invert


class Node(nn.Module):
    def __init__(self, prev, module):
        super(Node, self).__init__()
        # Always make into List
        if prev is not None:
            if not hasattr(prev, "__len__"):
                prev = [prev]
            prev = nn.ModuleList(prev)
        self.prev = prev
        self.next = []
        if self.prev is not None:
            for p in self.prev:
                p.register_next(self)
        self.module = module
        self.cur_out = None
        self.cur_in = None

    def register_next(self, next_module):
        self.next.append(next_module)

    def forward(self, x):
        out = self._forward(x)
        self.remove_cur_out()
        return out

    def _forward(self, x):
        if self.cur_out is None:
            # Collect incoming results
            if self.prev is not None:
                xs = [p.forward(x) for p in self.prev]
            else:
                xs = [x]
            x = self.module(*xs)
            self.cur_out = x
        return self.cur_out

    def invert(self, x):
        # determine starting module
        cur_ps = [self]
        starting_m = None
        while starting_m is None:
            new_cur_ps = []
            for p in cur_ps:
                if p.prev is None:
                    starting_m = p
                    break
                else:
                    new_cur_ps.extend(p.prev)
            cur_ps = new_cur_ps
        inverted = starting_m._invert(x)
        self.remove_cur_in()
        return inverted

    def _invert(self, y):
        if self.cur_in is None:
            # Collect incoming results
            if len(self.next) > 0:
                ys = []
                for n in self.next:
                    this_y = n._invert(y)
                    # Only take those ys belonging to you
                    if len(this_y) > 1 and len(n.prev) > 1:
                        assert len(this_y) == len(n.prev)
                        filtered_y = []
                        for p, a_y in zip(n.prev, this_y):
                            if p == self:
                                filtered_y.append(a_y)
                        this_y = filtered_y
                        if len(this_y) == 1:
                            this_y = this_y[0]
                    ys.append(this_y)

                if len(ys) == 1:
                    ys = ys[0]
            else:
                ys = y
            # wrap into sequential if necessary
            module = self.module
            if module.__class__.__name__ != nn.Sequential.__name__:

                module = nn.Sequential(module)
            x = invert(module, ys)
            self.cur_in = x
        return self.cur_in

    def remove_cur_out(self,):
        if self.prev is not None:
            for p in self.prev:
                p.remove_cur_out()
        self.cur_out = None

    def remove_cur_in(self,):
        if self.prev is not None:
            for p in self.prev:
                p.remove_cur_in()
        self.cur_in = None
