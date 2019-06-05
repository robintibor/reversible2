import numpy as np
import torch as th
import ot
from reversible2.util import ensure_on_same_device, np_to_var, var_to_np


def ot_euclidean_loss_for_samples(samples_a, samples_b):
    diffs = samples_a.unsqueeze(1) - samples_b.unsqueeze(0)
    diffs = th.norm(diffs, dim=2, p=2)
    return ot_for_diffs(diffs)

def ot_for_diffs(diffs):
    transport_mat = ot.emd([], [], var_to_np(diffs))
    # sometimes weird low values, try to prevent them
    # 0.5 could be 1.0, just set ot 0.5 to make more sure nothing
    # removed accidentally
    transport_mat = transport_mat * (transport_mat >= (0.5 / (diffs.numel())))

    transport_mat = np_to_var(transport_mat, dtype=np.float32,
                              device=diffs.device)
    loss = th.sum(transport_mat * diffs)
    return loss


def ot_euclidean_loss_memory_saving_for_samples(samples_a, samples_b):
    assert len(samples_a) <= len(samples_b)
    with th.no_grad():
        diffs = samples_a.unsqueeze(1) - samples_b.unsqueeze(0)
        diffs = th.sqrt(th.clamp(th.sum(diffs * diffs, dim=2), min=1e-6))
    transport_mat = ot.emd([], [], var_to_np(diffs))
    transport_mat = transport_mat * (transport_mat >= (0.5 / (diffs.numel())))
    del diffs

    b_corresponding = []
    for i_row in range(transport_mat.shape[0]):
        matches = np.flatnonzero(transport_mat[i_row])
        b_corresponding.append(samples_b[matches])
    b_corresponding = th.stack(b_corresponding, dim=0)
    loss = th.mean(
        th.norm(samples_a.unsqueeze(1) - b_corresponding, dim=2, p=2))
    return loss
