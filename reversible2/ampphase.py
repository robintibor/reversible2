import os
os.sys.path.insert(0, '/home/schirrmr/code/explaining/reversible//')
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/braindecode/')
from reversible.gaussian import get_gauss_samples
from reversible.uniform import get_uniform_samples
from reversible.revnet import SubsampleSplitter, invert
import numpy as np
import torch as th


def to_amp_phase(x, y):
    amps = th.sqrt((x * x) + (y * y))
    phases = th.atan2(y, x)
    return amps, phases


def outs_to_amp_phase(out):
    assert out.shape[2] == 2
    #n_features = out.shape[1]
    #assert n_features % 2  == 0
    x = out[:, :, 0::2]
    y = out[:, :, 1::2]
    return to_amp_phase(x,y)


def amp_phase_to_x_y(amps, phases):
    x, y = th.cos(phases), th.sin(phases)

    x = x * amps
    y = y * amps
    return x, y


def get_transformed_amp_phase_xy_samples(n_samples, mean, std,
                                         phase_dist,
                                         truncate_to,
                                      dtf_transform, transform_amps=True):
    amps, phases = get_amp_phase_samples(n_samples, mean, std,
                                         phase_dist=phase_dist,
                                         truncate_to=truncate_to)
    if transform_amps:
        amps, phases = th.chunk(
            dtf_transform(th.cat((amps, phases), dim=1)), 2, dim=1)
        #amps = th.abs(amps)
    else:
        phases = dtf_transform(phases)
    x, y = amp_phase_to_x_y(amps, phases)
    samples = interleave_x_y(x,y)#th.cat((x,y), dim=1)
    return samples


def get_amp_phase_samples(n_samples, mean, std, phase_dist, truncate_to):
    assert phase_dist in ["gauss", "uni"]
    i_half = len(mean) // 2

    amps = get_gauss_samples(n_samples, mean[:i_half], std[:i_half],
                             truncate_to=truncate_to)
    #amps = th.abs(amps)
    if phase_dist == 'uni':
        phases = get_uniform_samples(n_samples, mean[i_half:],
                                     std[i_half:] * 2 * np.pi)
    else:
        assert phase_dist == 'gauss'
        phases = get_gauss_samples(n_samples, mean[i_half:],
                                     std[i_half:] * 0.5 * np.pi,
                             truncate_to=truncate_to * 0.5 * np.pi)

    return amps, phases

def get_amp_phase_xy_samples(n_samples, mean, std, phase_dist, truncate_to):
    amps, phases = get_amp_phase_samples(n_samples, mean, std,
                   phase_dist=phase_dist, truncate_to=truncate_to)
    x, y = amp_phase_to_x_y(amps, phases)
    samples = interleave_x_y(x,y)#th.cat((x,y), dim=1)
    return samples

def interleave_x_y(x,y):
    assert x.shape == y.shape
    while len(x.shape) < 4:
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
    new_shape = list(x.shape)
    new_shape[2] *= 2
    x_y = th.zeros(new_shape, device=x.device)
    x_y[:,:,0::2] = x
    x_y[:,:,1::2] = y
    return x_y

def amp_phase_sample_to_x_y(amp_phase_sample):
    n_features = amp_phase_sample.shape[1]
    assert n_features % 2  == 0
    i_half = n_features // 2
    x,y = amp_phase_to_x_y(amp_phase_sample[:,:i_half], amp_phase_sample[:,i_half:])
    x_y = interleave_x_y(x,y)
    return x_y


def invert_to_interleaved_x_y(splitted,):
    splitter = SubsampleSplitter(stride=[2,1],chunk_chans_first=True)
    return invert(splitter, splitted)

def split_x_y_samples(samples):
    splitter = SubsampleSplitter(stride=[2,1], chunk_chans_first=True)
    samples = splitter(samples)
    return samples
