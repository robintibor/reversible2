import logging
from collections import OrderedDict

import numpy as np

from braindecode.datasets.bbci import BBCIDataset
from braindecode.mne_ext.signalproc import mne_apply
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import (
    exponential_running_standardize,
    bandpass_cnt,
)
from reversible2.util import np_to_var

log = logging.getLogger(__name__)


def load_file(filename, car=True, load_sensor_names=None):
    cnt = BBCIDataset(filename, load_sensor_names=load_sensor_names).load()
    cnt = cnt.drop_channels(["STI 014"])

    if car:

        def car(a):
            return a - np.mean(a, keepdims=True, axis=0)

        cnt = mne_apply(car, cnt)
    return cnt


def preprocess_cnt(cnt, final_hz, half_before):
    log.info("Resampling...")
    cnt = resample_cnt(cnt, 250.0)
    log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(
            a.T, factor_new=1e-3, init_block_size=1000, eps=1e-4
        ).T,
        cnt,
    )
    if half_before:
        cnt = resample_cnt(cnt, final_hz / 2.0)
    cnt = resample_cnt(cnt, final_hz)
    return cnt


def create_set(cnt, start_ms, stop_ms):
    marker_def = OrderedDict(
        [("Right Hand", [1]), ("Left Hand", [2]), ("Rest", [3]), ("Feet", [4])]
    )
    ival = [start_ms, stop_ms]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    return dataset


def create_inputs(cnt, final_hz, half_before, start_ms, stop_ms):
    cnt = preprocess_cnt(cnt, final_hz=final_hz, half_before=half_before)
    dataset = create_set(cnt, start_ms, stop_ms)
    return create_th_inputs(dataset)


def create_th_inputs(dataset):
    x_right = dataset.X[dataset.y == 0]
    x_rest = dataset.X[dataset.y == 2]
    inputs_a = np_to_var(x_right[:, :, :, None], dtype=np.float32)
    inputs_b = np_to_var(x_rest[:, :, :, None], dtype=np.float32)
    inputs = [inputs_a, inputs_b]
    return inputs


def load_train_test(
    subject_id,
    car,
    n_sensors,
    final_hz,
    start_ms,
    stop_ms,
    half_before,
    only_load_given_sensors,
):

    assert n_sensors in [2, 22]
    if n_sensors == 2:
        sensor_names = ["C3", "C4"]
    else:
        assert n_sensors == 22
        # fmt: off
        sensor_names = [
            'Fz',
            'FC3','FC1','FCz','FC2','FC4',
            'C5','C3','C1','Cz','C2','C4','C6',
            'CP3','CP1','CPz','CP2','CP4',
            'P1','Pz','P2',
            'POz']
        # fmt: on
    assert n_sensors == len(sensor_names)
    if only_load_given_sensors:
        load_sensor_names = sensor_names
    else:
        load_sensor_names = None

    orig_train_cnt = load_file(
        "/data/schirrmr/schirrmr/HGD-public/reduced/train/{:d}.mat".format(
            subject_id
        ),
        load_sensor_names=load_sensor_names,
        car=car,
    )
    train_cnt = orig_train_cnt.reorder_channels(sensor_names)

    train_inputs = create_inputs(
        train_cnt,
        final_hz=final_hz,
        half_before=half_before,
        start_ms=start_ms,
        stop_ms=stop_ms,
    )
    test_inputs = [t[-40:] for t in train_inputs]
    train_inputs = [t[:-40] for t in train_inputs]
    return train_inputs, test_inputs
