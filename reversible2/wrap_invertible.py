import torch
from torch import nn
import torch as th


class WrapInvertible(nn.Module):
    def __init__(self, module, keep_input=False, keep_output=False,
                 grad_is_inverse=False):
        super(WrapInvertible, self).__init__()
        self.module = module
        self.module.ctx_dict = dict()
        self.keep_input = keep_input
        self.keep_output = keep_output
        self.grad_is_inverse = grad_is_inverse

    def forward(self, x):
        grad_enabled = torch.is_grad_enabled()
        args = [x, self.module, grad_enabled, self.keep_input, self.keep_output,
                self.grad_is_inverse]
        out = WrapInvertibleFunction.apply(*args)
        return out

    def invert(self, y):
        grad_enabled = torch.is_grad_enabled()
        args = [y, self.module, grad_enabled, self.keep_input, self.keep_output,
                self.grad_is_inverse]
        out = WrapInvertibleInverseFunction.apply(*args)
        return out

def copy_first_if_needed(a, b):
    if a.data_ptr() == b.data_ptr():
        a = a.clone()
    return a


class WrapInvertibleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, grad_enabled, keep_input, keep_output, grad_is_inverse):
        ctx.module = module
        ctx.keep_input = keep_input
        ctx.keep_output = keep_output
        ctx.grad_is_inverse = grad_is_inverse
        with th.no_grad():
            out = module(x)
            # Make clone if storage is same
            out = copy_first_if_needed(out, x)
            if not keep_input:
                x.set_()
        if grad_enabled:
            module.ctx_dict[id(ctx)] = {}
            save_for_backward = []
            if not keep_input:
                module.ctx_dict[id(ctx)]['x'] = x
            if keep_output:
                save_for_backward.append(out)
            else:
                module.ctx_dict[id(ctx)]['output'] = out
            ctx.save_for_backward(*save_for_backward)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve input and output references
        if ctx.keep_output:
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
        if not ctx.keep_output:
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
        return (grad_input, None, None, None, None, None)


class WrapInvertibleInverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, module, grad_enabled, keep_input, keep_output, grad_is_inverse):
        ctx.module = module
        ctx.keep_input = keep_input
        ctx.keep_output = keep_output
        ctx.grad_is_inverse = grad_is_inverse
        with th.no_grad():
            x = module.invert(y)
            x = copy_first_if_needed(x, y)
            if not keep_output:
                y.set_()

        if grad_enabled:
            module.ctx_dict[id(ctx)] = {}
            save_for_backward = []
            if keep_input:
                save_for_backward.append(x)
            else:
                module.ctx_dict[id(ctx)]['x'] = x
            if not keep_output:
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
        if not ctx.keep_output:
            output = ctx.module.ctx_dict[id(ctx)].pop('output')

        if (not ctx.keep_output) or (not ctx.grad_is_inverse):
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
        if not ctx.keep_output:
            # in this case input was deleted so restore it
            output.set_(y)
        ctx.module.ctx_dict[id(ctx)].clear()
        return (grad_input, None, None, None, None, None)


class WrapInvertibleWithParams(nn.Module):
    def __init__(self, module, keep_input=False, keep_output=False):
        super(WrapInvertibleWithParams, self).__init__()
        self.F = module
        self.F.ctx_dict = {}
        self.keep_input = keep_input
        self.keep_output = keep_output

    def forward(self, x):
        grad_enabled = torch.is_grad_enabled()
        args = [x, grad_enabled, self.keep_input, self.keep_output,
                self.F, ] + list(self.F.parameters())

        out = WrapInvertibleWithParamsFunction.apply(*args)
        return out

    def invert(self, y):
        grad_enabled = torch.is_grad_enabled()
        args = [y, grad_enabled, self.keep_input, self.keep_output,
                self.F, ] + list(self.F.parameters())

        x = WrapInvertibleWithParamsInverseFunction.apply(*args)
        return x


class WrapInvertibleWithParamsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grad_enabled, keep_input, keep_output, F, *weights):
        assert hasattr(F, 'invert')
        # check if possible to partition into two equally sized partitions
        assert (x.shape[1] % 2 == 0)  # assert if proper split is possible
        # store partition size, Fm and Gm functions in context
        ctx.F = F
        ctx.keep_input = keep_input
        ctx.keep_output = keep_output

        with torch.no_grad():
            output = F(x)
            if not keep_input:
                x.set_()

                # save the (empty) input and (non-empty) output variables
        # do not use save_for_backward to avoid the checks
        # since it will be inplace-modified
        # ... now memory leak :(
        if grad_enabled:
            save_for_backward = []
            F.ctx_dict[id(ctx)] = {}
            if keep_input:
                save_for_backward.append(x)
            else:
                F.ctx_dict[id(ctx)]['x'] = x
            if keep_output:
                save_for_backward.append(output)
            else:
                F.ctx_dict[id(ctx)]['output'] = output
            ctx.save_for_backward(*save_for_backward)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve weight references
        F = ctx.F

        # retrieve input and output references
        if (ctx.keep_output) and (ctx.keep_input):
            x, output = ctx.saved_tensors
        elif (ctx.keep_output):
            output, = ctx.saved_tensors
        elif ctx.keep_input:
            x, = ctx.saved_tensors
        if not ctx.keep_output:
            output = F.ctx_dict[id(ctx)].pop('output')

        if not ctx.keep_input:
            # recompute x
            with torch.no_grad():
                x = F.invert(output)
                output.set_()
                del output

        with torch.set_grad_enabled(True):
            # compute outputs building a sub-graph
            x.requires_grad = True
            y = F(x)
            # perform full backward pass on graph...
            dd = torch.autograd.grad(y, (x,) + tuple(F.parameters()),
                                     grad_output)

        with torch.no_grad():
            FWgrads = dd[1:]
            grad_input = dd[0]

            if not ctx.keep_input:
                old_x = F.ctx_dict[id(ctx)].pop('x')
                # restore input
                # Will also restore it for the previous module in the chain!!
                # This makes backward possible
                old_x.set_(x.contiguous())
        F.ctx_dict[id(ctx)].clear()

        return (grad_input, None, None, None, None) + FWgrads


class WrapInvertibleWithParamsInverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, grad_enabled, keep_input, keep_output, F, *weights):
        # check if possible to partition into two equally sized partitions

        assert (y.shape[1] % 2 == 0)  # assert if proper split is possible
        # store partition size, Fm and Gm functions in context
        ctx.F = F
        ctx.keep_input = keep_input
        ctx.keep_output = keep_output

        with torch.no_grad():
            # partition in two equally sized set of channels
            # compute outputs
            x = F.invert(y)
            if not ctx.keep_output:
                y.set_()  # robintibor@gmail.com

        if grad_enabled:
            # save the (empty) input and (non-empty) output variables
            # do not use save_for_backward to avoid the checks
            # since it will be inplace-modified
            save_for_backward = []
            F.ctx_dict[id(ctx)] = {}
            if keep_input:
                save_for_backward.append(x)
            else:
                F.ctx_dict[id(ctx)]['x'] = x
            if keep_output:
                save_for_backward.append(y)
            else:
                F.ctx_dict[id(ctx)]['output'] = y
            ctx.save_for_backward(*save_for_backward)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve weight references
        F = ctx.F

        # retrieve input and output references
        if (ctx.keep_output) and (ctx.keep_input):
            x, output = ctx.saved_tensors
        elif (ctx.keep_output):
            output, = ctx.saved_tensors
        elif ctx.keep_input:
            x, = ctx.saved_tensors

        if not ctx.keep_input:
            x = F.ctx_dict[id(ctx)].pop('x')

        if not ctx.keep_output:
            output = F.ctx_dict[id(ctx)].pop('output')

        # reconstruct the output

        if not ctx.keep_output:
            # recompute xc
            with torch.no_grad():
                y = F(x)
                if not ctx.keep_input:
                    x.set_()
                    del x
        else:
            y = output

        F.ctx_dict[id(ctx)].clear()

        with torch.set_grad_enabled(True):
            # compute outputs building a sub-graph
            y.requires_grad = True

            x = F.invert(y)

            # perform full backward pass on graph...

            dd = torch.autograd.grad(x,
                                     (y,) + tuple(F.parameters()), grad_output)

        with torch.no_grad():
            FWgrads = dd[1:]
            grad_input = dd[0]

            if not ctx.keep_output:
                # restore input
                # Will also restore it for the previous module in the chain!!
                # This makes backward possible
                output.set_(y.contiguous())

        return (grad_input, None, None, None, None,) + FWgrads

