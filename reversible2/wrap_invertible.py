import torch
from torch import nn
import torch as th


class WrapInvertible(nn.Module):
    def __init__(self, module, keep_input=False, final_block=False,
                 grad_is_inverse=False):
        super(WrapInvertible, self).__init__()
        self.module = module
        self.module.ctx_dict = dict()
        self.keep_input = keep_input
        self.final_block = final_block
        self.grad_is_inverse = grad_is_inverse

    def forward(self, x):
        args = [x, self.module, self.keep_input, self.final_block,
                self.grad_is_inverse]
        out = WrapInvertibleFunction.apply(*args)
        return out

    def invert(self, y):
        args = [y, self.module, self.keep_input, self.final_block,
                self.grad_is_inverse]
        out = WrapInvertibleInverseFunction.apply(*args)
        return out

def copy_first_if_needed(a, b):
    if a.data_ptr() == b.data_ptr():
        a = a.clone()
    return a


class WrapInvertibleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, keep_input, final_block, grad_is_inverse):
        ctx.module = module
        ctx.keep_input = keep_input
        ctx.final_block = final_block
        ctx.grad_is_inverse = grad_is_inverse
        with th.no_grad():
            out = module(x)
            # Make clone if storage is same
            out = copy_first_if_needed(out, x)
            if not keep_input:
                x.set_()
        module.ctx_dict[id(ctx)] = {}
        save_for_backward = []
        if not keep_input:
            module.ctx_dict[id(ctx)]['x'] = x
        if final_block:
            save_for_backward.append(out)
        else:
            module.ctx_dict[id(ctx)]['output'] = out
        ctx.save_for_backward(*save_for_backward)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve input and output references
        if ctx.final_block:
            output, = ctx.saved_tensors
        else:
            output = ctx.module.ctx_dict[id(ctx)].pop('output')
        if not ctx.keep_input:
            x = ctx.module.ctx_dict[id(ctx)].pop('x')

        if (not ctx.keep_input) or (not ctx.grad_is_inverse):
            with th.no_grad():
                # in any case invertred will later be set to x (!)
                # so always needed
                inverted = ctx.module.invert(output)
                #inverted = copy_first_if_needed(inverted, output)
        if not ctx.final_block:
            output.set_()

        # Either directly compute grad as inverse
        # or recompute forward and then compute gradient
        if ctx.grad_is_inverse:
            with th.no_grad():
                grad_input = ctx.module.invert(grad_output)
        else:
            with torch.set_grad_enabled(True):
                inverted.requires_grad = True
                y = ctx.module(inverted)
                dd = torch.autograd.grad(y, (inverted,), grad_output)
            grad_input = dd[0]
        if not ctx.keep_input:
            # in this case input was deleted so restore it
            x.set_(inverted)
        ctx.module.ctx_dict[id(ctx)].clear()
        return (grad_input, None, None, None, None)


class WrapInvertibleInverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, module, keep_input, final_block, grad_is_inverse):
        ctx.module = module
        ctx.keep_input = keep_input
        ctx.final_block = final_block
        ctx.grad_is_inverse = grad_is_inverse
        with th.no_grad():
            x = module.invert(y)
            x = copy_first_if_needed(x, y)
            if not final_block:
                y.set_()
        module.ctx_dict[id(ctx)] = {}
        save_for_backward = []


        if keep_input:
            save_for_backward.append(x)
        else:
            module.ctx_dict[id(ctx)]['x'] = x
        if not final_block:
            module.ctx_dict[id(ctx)]['output'] = y
        ctx.save_for_backward(*save_for_backward)

        return x

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve input and output references
        if ctx.keep_input:
            x, = ctx.saved_tensors
        else:
            x = ctx.module.ctx_dict[id(ctx)].pop('x')
        if not ctx.final_block:
            output = ctx.module.ctx_dict[id(ctx)].pop('output')

        if (not ctx.final_block) or (not ctx.grad_is_inverse):
            with th.no_grad():
                y = ctx.module(x)
        if not ctx.keep_input:
            x.set_()
        # Either directly compute grad as inverse
        # or recompute forward and then compute gradient
        if ctx.grad_is_inverse:
            grad_input = ctx.module(grad_output)
        else:
            with torch.set_grad_enabled(True):
                y.requires_grad = True
                inverted = ctx.module.invert(y)
                dd = torch.autograd.grad(inverted, (y,), grad_output)
            grad_input = dd[0]
        if not ctx.final_block:
            # in this case input was deleted so restore it
            output.set_(y)
        ctx.module.ctx_dict[id(ctx)].clear()
        return (grad_input, None, None, None, None)
