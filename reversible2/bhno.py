import logging
from collections import OrderedDict

import numpy as np

from braindecode.datasets.bbci import BBCIDataset
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import exponential_running_standardize, bandpass_cnt
from reversible2.util import np_to_var

log = logging.getLogger(__name__)

def load_file(filename):
    cnt = BBCIDataset(filename).load()
    cnt = cnt.drop_channels(['STI 014'])
    def car(a):
        return a - np.mean(a, keepdims=True, axis=0)

    cnt = mne_apply(
        car, cnt)
    return cnt


def preprocess_cnt(cnt, final_hz, half_before):
    log.info("Resampling train...")
    cnt = resample_cnt(cnt, 250.0)
    log.info("Standardizing train...")
    cnt = mne_apply(lambda a: exponential_running_standardize(a.T ,factor_new=1e-3, init_block_size=1000, eps=1e-4).T,
                         cnt)
    if half_before:
        cnt = resample_cnt(cnt, final_hz / 2.0)
    cnt = resample_cnt(cnt, final_hz)
    return cnt


def create_set(cnt):
    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                             ('Rest', [3]), ('Feet', [4])])
    ival = [500,1500]


    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    return dataset


def create_inputs(cnt, final_hz, half_before):
    cnt = preprocess_cnt(cnt, final_hz=final_hz, half_before=half_before)
    dataset = create_set(cnt)
    return create_th_inputs(dataset)


def create_th_inputs(dataset):
    x_right = dataset.X[dataset.y == 0]
    x_rest = dataset.X[dataset.y == 2]
    inputs_a = np_to_var(x_right[:,0:1,:,None], dtype=np.float32)
    inputs_b = np_to_var(x_rest[:,0:1,:,None], dtype=np.float32)
    inputs = [inputs_a, inputs_b]
    return inputs
