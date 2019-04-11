import torch as th
import numpy as np

# For truncated logic see:
#https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
# torch.fmod(torch.randn(size),2)
def get_gauss_samples(n_samples, mean, std, truncate_to=None):
    if mean.is_cuda:
        orig_samples = th.cuda.FloatTensor(n_samples, len(mean)).normal_(0, 1)
    else:
        orig_samples = th.FloatTensor(n_samples, len(mean)).normal_(0, 1)
    if truncate_to is not None:
        orig_samples = th.fmod(orig_samples, truncate_to)
    orig_samples = th.autograd.Variable(orig_samples)
    samples = (orig_samples * std.unsqueeze(0)) + mean.unsqueeze(0)
    return samples

