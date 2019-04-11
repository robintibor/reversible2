import torch as th
import numpy as np
import os
os.sys.path.insert(0, '/home/schirrmr/code/explaining/reversible//')
from reversible.gaussian import get_gauss_samples

def sample_hierarchically(n_samples, mean, log_stds):
    cur_mean = mean
    covs = th.zeros((len(cur_mean), len(cur_mean)), dtype=th.float32)
    hierarchical_samples = []
    for i_exp in range(int(np.log2(len(cur_mean))) + 1):
        cur_mean = th.stack(th.chunk(cur_mean, int(2**i_exp)))
        this_mean = th.mean(cur_mean, dim=1, keepdim=True)
        cur_mean = cur_mean - this_mean
        cur_mean = cur_mean.view(-1)
        this_log_std = log_stds[i_exp]
        # sample...
        this_samples = get_gauss_samples(n_samples, this_mean.squeeze(-1), th.exp(this_log_std).squeeze(-1))
        hierarchical_samples.append(this_samples)
        # compute cov matrix
        for i_part in range(2 ** i_exp):
            i_1, i_2 = int((i_part/2**i_exp) * len(covs)), int(((i_part+1)/2**i_exp) * len(covs))
            covs[i_1:i_2, i_1:i_2] += (th.exp(this_log_std[i_part]) ** 2)
    samples = convert_hierarchical_samples_to_samples(hierarchical_samples, len(mean))
    return samples, covs

def convert_hierarchical_samples_to_samples(hierarchical_samples, n_dims):
    n_samples = len(hierarchical_samples[0])
    samples = th.zeros((n_samples, n_dims), dtype=th.float32)
    for i_exp in range(int(np.log2(n_dims)) + 1):
        this_samples = hierarchical_samples[i_exp]
        samples += this_samples.view(-1).repeat(
            n_dims // int(2**i_exp),1).t().contiguous().view(samples.shape)
    return samples


def convert_haar_wavelet(wavelet_samples):
    if len(wavelet_samples[0].shape) == 1:
        n_dims = wavelet_samples.shape[1]
        # first convert to wanted represetation,
        # list per hierarchy level
        proper_samples = [wavelet_samples[:, 0:1]]
        for i_exp in range(int(np.log2(n_dims))):
            proper_samples.append(
                wavelet_samples[:, int(2 ** i_exp):int(2 ** (i_exp + 1))])
        wavelet_samples = proper_samples
    n_dims = 2 ** (len(wavelet_samples) - 1)
    n_samples = len(wavelet_samples[0])

    samples = th.zeros((n_samples, n_dims), dtype=th.float32)
    samples = wavelet_samples[0] + samples
    for i_exp in range(int(np.log2(n_dims))):
        w_samples = wavelet_samples[i_exp + 1]
        n_repeats = int(n_dims / w_samples.shape[1])
        # repeat to fit samples e.g. repeat
        # [[2,4,-1,3]] (in this example, n_samples=1) for n-dims 16 to
        # [[2,2,2,2,4,4,4,4,-1,-1,-1,-1,3,3,3,3]]
        expanded_samples = w_samples.unsqueeze(-1).repeat(1, 1, n_repeats).view(
            n_samples, -1)
        wavelet_mask = th.cat(
            (-th.ones(n_repeats // 2), th.ones(n_repeats // 2)))
        wavelet_mask = wavelet_mask.repeat(w_samples.shape[1])
        samples = samples + expanded_samples * wavelet_mask.unsqueeze(0)
    return samples

def convert_wavelet(wavelet_samples, wavelet_mask_arr):
    if len(wavelet_samples[0].shape) == 1:
        n_dims = wavelet_samples.shape[1]
        # first convert to wanted represetation,
        # list per hierarchy level
        proper_samples = [wavelet_samples[:, 0:1]]
        for i_exp in range(int(np.log2(n_dims))):
            proper_samples.append(
                wavelet_samples[:, int(2 ** i_exp):int(2 ** (i_exp + 1))])
        wavelet_samples = proper_samples
    n_dims = 2 ** (len(wavelet_samples) - 1)
    n_samples = len(wavelet_samples[0])

    samples = th.zeros((n_samples, n_dims), dtype=th.float32)
    samples = wavelet_samples[0] + samples # add mean sample
    for i_exp in range(int(np.log2(n_dims))):
        w_samples = wavelet_samples[i_exp + 1]
        n_repeats = int(n_dims / w_samples.shape[1])
        # repeat to fit samples e.g. repeat
        # [[2,4,-1,3]] (in this example, n_samples=1) for n-dims 16 to
        # [[2,2,2,2,4,4,4,4,-1,-1,-1,-1,3,3,3,3]]
        expanded_samples = w_samples.unsqueeze(-1).repeat(1, 1, n_repeats).view(
            n_samples, -1)
        wavelet_mask = wavelet_mask_arr[i_exp]
        wavelet_mask = wavelet_mask - th.mean(wavelet_mask)
        wavelet_mask = wavelet_mask.repeat(w_samples.shape[1])
        samples = samples + expanded_samples * wavelet_mask.unsqueeze(0)
    return samples

def sample_wavelet(n_samples, mean, log_std, truncate_to=3,
                   convert_fn=convert_haar_wavelet):
    wavelet_samples = []
    # sample first one outside
    mean_sample = get_gauss_samples(
        n_samples, mean[0:1], th.exp(log_std[0:1]),
        truncate_to=truncate_to)
    wavelet_samples.append(mean_sample)
    for i_exp in range(int(np.log2(len(mean)))):
        i_start = int(2**i_exp)
        i_stop = int(2**(i_exp+1))
        this_mean = mean[i_start:i_stop]
        this_log_std = log_std[i_start:i_stop]
        this_samples = get_gauss_samples(
            n_samples, this_mean, th.exp(this_log_std), truncate_to=truncate_to)
        wavelet_samples.append(this_samples)
    return convert_fn(wavelet_samples)


def convert_linear_wavelet(wavelet_samples):
    if len(wavelet_samples[0].shape) == 1:
        n_dims = wavelet_samples.shape[1]
        # first convert to wanted represetation,
        # list per hierarchy level
        proper_samples = [wavelet_samples[:, 0:1]]
        for i_exp in range(int(np.log2(n_dims))):
            proper_samples.append(
                wavelet_samples[:, int(2 ** i_exp):int(2 ** (i_exp + 1))])
        wavelet_samples = proper_samples
    n_dims = 2 ** (len(wavelet_samples) - 1)
    n_samples = len(wavelet_samples[0])

    samples = th.zeros((n_samples, n_dims), dtype=th.float32)
    samples = wavelet_samples[0] + samples
    for i_exp in range(int(np.log2(n_dims))):
        w_samples = wavelet_samples[i_exp + 1]
        n_repeats = int(n_dims / w_samples.shape[1])
        # repeat to fit samples e.g. repeat
        # [[2,4,-1,3]] (in this example, n_samples=1) for n-dims 16 to
        # [[2,2,2,2,4,4,4,4,-1,-1,-1,-1,3,3,3,3]]
        expanded_samples = w_samples.unsqueeze(-1).repeat(1, 1, n_repeats).view(
            n_samples, -1)
        wavelet_mask = th.linspace(-2 + (2 / n_repeats), 2 - (2 / n_repeats),
                                   n_repeats)
        wavelet_mask = wavelet_mask.repeat(w_samples.shape[1])
        samples = samples + expanded_samples * wavelet_mask.unsqueeze(0)
    return samples
