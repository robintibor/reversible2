import numpy as np
import torch as th
import ot
from reversible.util import ensure_on_same_device, np_to_var, var_to_np

def ot_euclidean_loss_for_samples(samples_a, samples_b):
    diffs = samples_a.unsqueeze(1) - samples_b.unsqueeze(0)
    diffs = th.sqrt(th.clamp(th.sum(diffs * diffs, dim=2), min=1e-6))

    transport_mat = ot.emd([], [], var_to_np(diffs))
    # sometimes weird low values, try to prevent them
    transport_mat = transport_mat * (transport_mat > (1.0/(diffs.numel())))

    transport_mat = np_to_var(transport_mat, dtype=np.float32)
    diffs, transport_mat = ensure_on_same_device(diffs, transport_mat)
    loss = th.sum(transport_mat * diffs)
    return loss