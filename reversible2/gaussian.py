import torch as th
import numpy as np

# For truncated logic see:
# https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
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


def get_gaussian_log_probs(mean, log_std, outs):
    demeaned = outs - mean.unsqueeze(0)
    unnormed_log_probs = -(demeaned ** 2) / (2 * (th.exp(log_std) ** 2))
    log_probs = unnormed_log_probs - np.log(float(np.sqrt(2 * np.pi))) - log_std
    log_probs = th.sum(log_probs, dim=1)
    return log_probs

def transform_gaussians_by_dirs(means, stds, directions):
    # directions is directions x dims
    # means is clusters x dims
    # stds is clusters x dims
    transformed_means = th.mm(means, directions.transpose(1, 0)).transpose(1, 0)
    # transformed_means is now
    # directions x clusters
    stds_for_dirs = stds.transpose(1, 0).unsqueeze(0)  # 1 x dims x clusters
    transformed_stds = th.sqrt(th.sum(
        (directions * directions).unsqueeze(2) *
        (stds_for_dirs * stds_for_dirs),
        dim=1))
    # transformed_stds is now
    # directions x clusters
    # so switch both back to clusters x directions
    return transformed_means.t(), transformed_stds.t()