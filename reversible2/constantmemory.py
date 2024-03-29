import torch
import torch.nn as nn
import copy
from copy import deepcopy
from reversible2.wrap_invertible import WrapInvertible, WrapInvertibleWithParams


class AdditiveBlockConstantMemory(nn.Module):
    def __init__(
        self,
        Fm,
        Gm=None,
        implementation_fwd=0,
        implementation_bwd=0,
        keep_input=False,
        keep_output=False,
    ):
        """The AdditiveBlock
        Parameters
        ----------
            Fm : torch.nn.Module
                A torch.nn.Module encapsulating an arbitrary function
            Gm : torch.nn.Module
                A torch.nn.Module encapsulating an arbitrary function
                (If not specified a deepcopy of Gm is used as a Module)
            implementation_fwd : int
                Switch between different Additive Operation implementations for forward pass. Default = 1
            implementation_bwd : int
                Switch between different Additive Operation implementations for inverse pass. Default = 1
        """
        super(AdditiveBlockConstantMemory, self).__init__()
        # mirror the passed module, without parameter sharing...
        if Gm is None:
            Gm = copy.deepcopy(Fm)
        self.Gm = Gm
        self.Fm = Fm
        self.Fm.ctx_dict = dict()
        self.implementation_fwd = implementation_fwd
        self.implementation_bwd = implementation_bwd
        self.keep_input = keep_input
        self.keep_output = keep_output

    def forward(self, x):
        grad_enabled = torch.is_grad_enabled()
        args = (
            [
                x,
                grad_enabled,
                self.keep_input,
                self.keep_output,
                self.Fm,
                self.Gm,
            ]
            + [w for w in self.Fm.parameters()]
            + [w for w in self.Gm.parameters()]
        )

        if self.implementation_fwd == 0:
            out = AdditiveBlockFunction.apply(*args)
        else:
            raise NotImplementedError(
                "Selected implementation ({}) not implemented...".format(
                    self.implementation_fwd
                )
            )

        return out

    def invert(self, y):
        grad_enabled = torch.is_grad_enabled()
        args = (
            [
                y,
                grad_enabled,
                self.keep_input,
                self.keep_output,
                self.Fm,
                self.Gm,
            ]
            + [w for w in self.Fm.parameters()]
            + [w for w in self.Gm.parameters()]
        )

        if self.implementation_bwd == 0:
            x = AdditiveBlockInverseFunction.apply(*args)
        else:
            raise NotImplementedError(
                "Inverse for selected implementation ({}) not implemented...".format(
                    self.implementation_bwd
                )
            )

        return x


class AdditiveBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, grad_enabled, keep_input, keep_output, Fm, Gm, *weights
    ):
        """Forward pass for the reversible block computes:
        {x1, x2} = x
        y1 = x1 + Fm(x2)
        y2 = x2 + Gm(y1)
        output = {y1, y2}
        Parameters
        ----------
        ctx : torch.autograd.function.AdditiveBlockFunctionBackward
            The backward pass context object
        x : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}
        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this function
        """
        # check if possible to partition into two equally sized partitions
        assert x.shape[1] % 2 == 0  # assert if proper split is possible
        # store partition size, Fm and Gm functions in context
        ctx.Fm = Fm
        ctx.Gm = Gm
        ctx.keep_input = keep_input
        ctx.keep_output = keep_output

        with torch.no_grad():
            # partition in two equally sized set of channels
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1, x2 = x1.contiguous(), x2.contiguous()

            # compute outputs
            fmr = Fm.forward(x2)

            y1 = x1 + fmr
            x1.set_()
            del x1
            gmr = Gm.forward(y1)
            y2 = x2 + gmr
            x2.set_()
            del x2
            output = torch.cat([y1, y2], dim=1)
            y1.set_()
            y2.set_()
            del y1, y2
            if not keep_input:
                x.set_()  # robintibor@gmail.com

        # save the (empty) input and (non-empty) output variables
        # do not use save_for_backward to avoid the checks
        # since it will be inplace-modified
        # ... now memory leak :(
        if grad_enabled:
            save_for_backward = []

            Fm.ctx_dict[id(ctx)] = {}
            if keep_input:
                save_for_backward.append(x)
            else:
                Fm.ctx_dict[id(ctx)]["x"] = x
            if keep_output:
                save_for_backward.append(output)
            else:
                Fm.ctx_dict[id(ctx)]["output"] = output
            ctx.save_for_backward(*save_for_backward)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve weight references
        Fm, Gm = ctx.Fm, ctx.Gm

        # retrieve input and output references
        if (ctx.keep_output) and (ctx.keep_input):
            x, output = ctx.saved_tensors
        elif ctx.keep_output:
            output, = ctx.saved_tensors
        elif ctx.keep_input:
            x, = ctx.saved_tensors
        if not ctx.keep_output:
            output = Fm.ctx_dict[id(ctx)].pop("output")

        GWeights = [p for p in Gm.parameters()]
        if not ctx.keep_input:
            # recompute x
            with torch.no_grad():
                y1, y2 = torch.chunk(output, 2, dim=1)
                output.set_()
                del output
                y1, y2 = y1.contiguous(), y2.contiguous()

                # partition output gradient also on channels
                assert grad_output.shape[1] % 2 == 0

                x2 = y2 - Gm.forward(y1)
                x1 = y1 - Fm.forward(x2)
                y2.set_()
                y1.set_()
                del y1, y2
        else:
            x1, x2 = torch.chunk(x, 2, dim=1)

        with torch.set_grad_enabled(True):
            # compute outputs building a sub-graph
            x1.requires_grad = True
            x2.requires_grad = True

            y1 = x1 + Fm.forward(x2)
            y2 = x2 + Gm.forward(y1)
            y = torch.cat([y1, y2], dim=1)

            # perform full backward pass on graph...
            dd = torch.autograd.grad(
                y,
                (x1, x2) + tuple(Gm.parameters()) + tuple(Fm.parameters()),
                grad_output,
            )

        with torch.no_grad():
            GWgrads = dd[2 : 2 + len(GWeights)]
            FWgrads = dd[2 + len(GWeights) :]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)

            # cleanup sub-graph
            y1.detach_()
            y2.detach_()
            y1.set_()
            y2.set_()
            del y1, y2

            if not ctx.keep_input:
                x = Fm.ctx_dict[id(ctx)].pop("x")
                # restore input
                # Will also restore it for the previous module in the chain!!
                # This makes backward possible
                x.set_(torch.cat([x1, x2], dim=1).contiguous())
        Fm.ctx_dict[id(ctx)].clear()

        return (grad_input, None, None, None, None, None) + FWgrads + GWgrads


class AdditiveBlockInverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, y, grad_enabled, keep_input, keep_output, Fm, Gm, *weights
    ):
        """Forward pass for the reversible block computes:
        {y1, y2} = y
        x2 = y2 - Gm(y1)
        x1 = y1 - Fm(x2)
        output = {x1, x2}
        Parameters
        ----------
        cty : torch.autograd.function.RevNetInverseFunctionBackward
            The backward pass context object
        y : TorchTensor
            Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
        Fm : nn.Module
            Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
        Gm : nn.Module
            Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
        *weights : TorchTensor
            weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}
        Note
        ----
        All tensor/autograd variable input arguments and the output are
        TorchTensors for the scope of this fuction
        """
        # check if possible to partition into two equally sized partitions

        assert y.shape[1] % 2 == 0  # assert if proper split is possible
        # store partition size, Fm and Gm functions in context
        ctx.Fm = Fm
        ctx.Gm = Gm
        ctx.keep_input = keep_input
        ctx.keep_output = keep_output

        with torch.no_grad():
            # partition in two equally sized set of channels
            y1, y2 = torch.chunk(y, 2, dim=1)
            y1, y2 = y1.contiguous(), y2.contiguous()

            # compute outputs

            x2 = y2 - Gm.forward(y1)
            y2.set_()
            del y2

            x1 = y1 - Fm.forward(x2)
            y1.set_()
            del y1

            x = torch.cat([x1, x2], dim=1)
            x1.set_()
            x2.set_()
            del x1, x2
            if not ctx.keep_output:
                y.set_()  # robintibor@gmail.com

        if grad_enabled:
            # save the (empty) input and (non-empty) output variables
            # do not use save_for_backward to avoid the checks
            # since it will be inplace-modified
            save_for_backward = []
            Fm.ctx_dict[id(ctx)] = {}
            if keep_input:
                save_for_backward.append(x)
            else:
                Fm.ctx_dict[id(ctx)]["x"] = x
            if keep_output:
                save_for_backward.append(y)
            else:
                Fm.ctx_dict[id(ctx)]["output"] = y
            ctx.save_for_backward(*save_for_backward)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve weight references
        Fm, Gm = ctx.Fm, ctx.Gm

        # retrieve input and output references
        if (ctx.keep_output) and (ctx.keep_input):
            x, output = ctx.saved_tensors
        elif ctx.keep_output:
            output, = ctx.saved_tensors
        elif ctx.keep_input:
            x, = ctx.saved_tensors

        if not ctx.keep_input:
            x = Fm.ctx_dict[id(ctx)].pop("x")

        if not ctx.keep_output:
            output = Fm.ctx_dict[id(ctx)].pop("output")

        # reconstruct the output

        GWeights = [p for p in Gm.parameters()]
        if not ctx.keep_output:
            # recompute xc
            with torch.no_grad():
                x1, x2 = torch.chunk(x, 2, dim=1)
                x.set_()
                del x
                x1, x2 = x1.contiguous(), x2.contiguous()

                # partition output gradient also on channels
                assert grad_output.shape[1] % 2 == 0

                y1 = x1 + Fm.forward(x2)
                y2 = x2 + Gm.forward(y1)
                x1.set_()
                x2.set_()
                del x1, x2
        else:
            y1, y2 = torch.chunk(output, 2, dim=1)
        Fm.ctx_dict[id(ctx)].clear()

        with torch.set_grad_enabled(True):
            # compute outputs building a sub-graph
            y1.requires_grad = True
            y2.requires_grad = True

            x2 = y2 - Gm.forward(y1)
            x1 = y1 - Fm.forward(x2)

            x = torch.cat([x1, x2], dim=1)

            # perform full backward pass on graph...

            dd = torch.autograd.grad(
                x,
                (y1, y2) + tuple(Gm.parameters()) + tuple(Fm.parameters()),
                grad_output,
            )

        with torch.no_grad():
            GWgrads = dd[2 : 2 + len(GWeights)]
            FWgrads = dd[2 + len(GWeights) :]
            grad_input = torch.cat([dd[0], dd[1]], dim=1)

            # cleanup sub-graph
            x1.detach_()
            x2.detach_()
            x1.set_()
            x2.set_()
            del x1, x2

            if not ctx.keep_output:
                # restore input
                # Will also restore it for the previous module in the chain!!
                # This makes backward possible
                output.set_(torch.cat([y1, y2], dim=1).contiguous())

        return (grad_input, None, None, None, None, None) + FWgrads + GWgrads


def clear_ctx_dicts(model):
    for m in model.modules():
        if hasattr(m, "ctx_dict"):
            m.ctx_dict.clear()


def sequential_to_constant_memory(seq_model):
    children = list(seq_model.children())
    new_children = []
    for i_c, c in enumerate(children):
        keep_input = i_c == 0
        keep_output = i_c == len(children) - 1
        new_c = module_to_constant_memory(c, keep_input, keep_output)
        new_children.append(new_c)
    new_seq_model = nn.Sequential(*new_children)
    assert new_seq_model[0].keep_input == True
    assert new_seq_model[-1].keep_output == True
    return new_seq_model


def module_to_constant_memory(c, keep_input, keep_output):
    c = deepcopy(c)
    classname = c.__class__.__name__
    if classname == "AdditiveBlock":
        assert c.switched_order == False
        return AdditiveBlockConstantMemory(
            c.FA, c.GA, keep_input=keep_input, keep_output=keep_output
        )
    elif classname in ["SubsampleSplitter", "ViewAs"]:
        return WrapInvertible(
            c,
            keep_input=keep_input,
            keep_output=keep_output,
            grad_is_inverse=True,
        )
    elif classname in ["RFFT", "IRFFT"]:
        return WrapInvertible(
            c,
            keep_input=keep_input,
            keep_output=keep_output,
            grad_is_inverse=False,
        )
    elif classname in ["ScalingLayer"]:
        return WrapInvertibleWithParams(
            c, keep_input=keep_input, keep_output=keep_output
        )
    else:
        raise ValueError(
            "Unknown class to convert to constant memory {:s}".format(classname)
        )


def graph_to_constant_memory(final_node):
    from reversible2.graph import get_all_nodes

    all_nodes = get_all_nodes(final_node)
    for n in all_nodes:
        if n.module.__class__.__name__ == nn.Sequential.__name__:
            n.module = sequential_to_constant_memory(n.module)
    return final_node
