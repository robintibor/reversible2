import os
#os.sys.path.insert(0, '/home/schirrmr/code/memcnn/')

import torch
import torch.nn as nn
import copy
import warnings
from memcnn.models.additive import AdditiveBlock



class ReversibleBlock(nn.Module):
    def __init__(self, Fm, Gm=None, coupling='additive', keep_input=False, keep_output=False,
                 implementation_fwd=0, implementation_bwd=0):
        """The ReversibleBlock

        Parameters
        ----------
            Fm : torch.nn.Module
                A torch.nn.Module encapsulating an arbitrary function

            Gm : torch.nn.Module
                A torch.nn.Module encapsulating an arbitrary function
                (If not specified a deepcopy of Fm is used as a Module)

            coupling: str
                Type of coupling ['additive', 'affine']. Default = 'additive'
                'affine' is currently experimental

            keep_input : bool
                Retain the input information, by default it can be discarded since it will be
                reconstructed upon the backward pass.

            implementation_fwd : int
                Switch between different Operation implementations for forward training. Default = 1
                -1 : Naive implementation without reconstruction on the backward pass (keep_input should be True)
                 0 : Memory efficient implementation, compute gradients directly on y
                 1 : Memory efficient implementation, similar to approach in Gomez et al. 2017

            implementation_bwd : int
                Switch between different Operation implementations for backward training. Default = 1
                -1 : Naive implementation without reconstruction on the backward pass (keep_input should be True)
                 0 : Memory efficient implementation, compute gradients directly on y
                 1 : Memory efficient implementation, similar to approach in Gomez et al. 2017

        """
        super(ReversibleBlock, self).__init__()
        self.keep_input = keep_input
        self.keep_output = keep_output
        if coupling == 'additive':
            self.rev_block = AdditiveBlock(Fm, Gm, implementation_fwd, implementation_bwd)
        elif coupling == 'affine':
            self.rev_block = AffineBlock(Fm, Gm, implementation_fwd, implementation_bwd)
        else:
            raise NotImplementedError('Unknown coupling method: %s' % coupling)

    def forward(self, x):
        y = self.rev_block(x)
        # clears the referenced storage data linked to the input tensor as it can be reversed on the backward pass
        if not self.keep_input:
            x.storage().resize_(0)

        return y

    def invert(self, y):
        x = self.rev_block.inverse(y)
        # clears the referenced storage data linked to the input tensor as it can be reversed on the backward pass
        if not self.keep_output:
            y.storage().resize_(0)

        return x

class AdditiveBlockConstantMemory(ReversibleBlock):
    def __init__(self, Fm, Gm=None, keep_input=False, keep_output=False,
                 implementation_fwd=0, implementation_bwd=0):
        super(AdditiveBlockConstantMemory, self).__init__(
        Fm=Fm, Gm=Gm, keep_input=keep_input, keep_output=keep_output,
        implementation_fwd=implementation_fwd,
            implementation_bwd=implementation_bwd)


def clear_ctx_dicts(model):
    for m in model.modules():
        if hasattr(m, 'ctx_dict'):
            m.ctx_dict.clear()