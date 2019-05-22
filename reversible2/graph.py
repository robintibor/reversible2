import torch as th
from torch import nn
from reversible2.invert import invert
from reversible2.constantmemory import clear_ctx_dicts


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
        # ps are predecessors
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

    def data_init(self, x):
        with th.no_grad():
            self._data_init(x)
            self.remove_cur_out()
        clear_ctx_dicts(self)
        return


    def _data_init(self, x):
        if self.cur_out is None:
            # Collect incoming results
            if self.prev is not None:
                xs = [p._data_init(x) for p in self.prev]
            else:
                xs = [x]
            if self.module.__class__.__name__ == nn.Sequential.__name__:
                assert len(xs) == 1
                x = xs[0]
                for child in self.module.children():
                    weight_params = [m.weight for m in child.modules() if
                                     hasattr(m, 'weight')]
                    if len(weight_params) > 0:
                        upper_bound = None
                        lower_bound = None
                        factor = 1.0
                        for i in range(10):
                            old_factor = factor
                            next_x = child(x.clone())
                            var_x = th.var(next_x, )
                            if i == 0:
                                first_var_x = var_x.item() # just for logging
                            if var_x.item() > 1:
                                upper_bound = factor
                                if (lower_bound is None):
                                    factor = factor / 2
                            else:
                                lower_bound = factor
                                if upper_bound is None:
                                    factor = factor * 2
                            if (upper_bound is not None) and (
                                    lower_bound is not None):
                                factor = (lower_bound + upper_bound) / 2
                            for p in weight_params:
                                p.data = p.data * (factor / old_factor)
                        next_x = child(x.clone())
                        final_var_x = th.var(next_x, ).item()
                        print("Changed variance from {:.2f} to {:.2f}".format(
                            first_var_x, final_var_x
                        ))
                    x = child(x)
            else:
                x = self.module(*xs)
            self.cur_out = x
        return self.cur_out


def data_init_sequential_module(module, x):
    with th.no_grad():
        for child in module.children():
            weight_params = [m.weight for m in child.modules() if
                             hasattr(m, 'weight')]
            if len(weight_params) > 0:
                upper_bound = None
                lower_bound = None
                factor = 1.0
                for i in range(10):
                    old_factor = factor
                    next_x = child(x.clone())
                    var_x = th.var(next_x, )
                    if i == 0:
                        first_var_x = var_x.item()  # just for logging
                    if var_x.item() > 1:
                        upper_bound = factor
                        if (lower_bound is None):
                            factor = factor / 2
                    else:
                        lower_bound = factor
                        if upper_bound is None:
                            factor = factor * 2
                    if (upper_bound is not None) and (
                            lower_bound is not None):
                        factor = (lower_bound + upper_bound) / 2
                    for p in weight_params:
                        p.data = p.data * (factor / old_factor)
                next_x = child(x.clone())
                final_var_x = th.var(next_x, ).item()

                print("Changed variance from {:.2f} to {:.2f}\nusing factor {:.1E}".format(
                    first_var_x, final_var_x, factor
                ))
            x = child(x)
    return x

