import os
import site

os.sys.path.insert(0, "/home/schirrmr/code/reversible/")
os.sys.path.insert(0, "/home/schirrmr/braindecode/code/braindecode/")
import logging
import time
import numpy as np
import torch as th
from torch import nn

from hyperoptim.parse import (
    cartesian_dict_of_lists_product,
    product_of_list_of_lists_of_dicts,
)
from hyperoptim.util import save_pkl_artifact
from braindecode.torch_ext.util import var_to_np, confirm_gpu_availability
from braindecode.torch_ext.modules import Expression


from reversible2.distribution import TwoClassDist, TwoClassIndependentDist
from reversible2.monitor import compute_accs
from reversible2.models import smaller_model, larger_model
from reversible2.training import CLFTrainer
from reversible2.classifier import SubspaceClassifier
from reversible2.monitor import compute_clf_accs
from reversible2.high_gamma import load_file, create_inputs, load_train_test
from reversible2.models import add_bnorm_before_relu
from reversible2.models import WrappedModel
from reversible2.models import deep_invertible
from reversible2.scale import scale_to_unit_var


log = logging.getLogger(__name__)
log.setLevel("DEBUG")

from braindecode.models.base import BaseModel

class InvertibleModel(BaseModel):
    def __init__(self, n_chans, n_time,final_fft, add_bnorm):
        self.n_chans = n_chans
        self.n_time =  n_time
        self.final_fft = final_fft
        self.add_bnorm = add_bnorm

    def create_network(self):
        feature_model = larger_model(
            self.n_chans, self.n_time, final_fft=self.final_fft, constant_memory=False
        )
        model = nn.Sequential(feature_model,
                              Expression(lambda x: x[:,:2]),
                              nn.LogSoftmax(dim=1))
        if self.add_bnorm:
            add_bnorm_before_relu(model)
        model.eval()
        return model


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [
        {
            "save_folder": "/data/schirrmr/schirrmr/reversible/experiments/deepshallow",
            "only_return_exp": False,
            "debug": False,
        }
    ]
    subject_id_params = dictlistprod({"subject_id": range(4, 10)})
    data_params = dictlistprod({"n_sensors": [22], "final_hz": [256]})
    preproc_params = dictlistprod({"half_before": [True]})
    ival_params = [{"start_ms": 500, "stop_ms": 1500}]
    training_params = dictlistprod({"max_epochs": [100]})

    model_params = dictlistprod({"model": ["deep_invertible",],
                                 "final_fft": [True],
                                 "add_bnorm": [False],})  # , True

    optim_params = dictlistprod({"weight_decay": [0.5 * 0.001, 0.5*0.01],
                                 "act_norm": [True, False]})

    save_params = [{"save_model": False}]

    grid_params = product_of_list_of_lists_of_dicts(
        [
            default_params,
            subject_id_params,
            data_params,
            training_params,
            preproc_params,
            ival_params,
            training_params,
            model_params,
            save_params,
            optim_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params


def run_exp(
    debug,
    subject_id,
    max_epochs,
    n_sensors,
    final_hz,
    half_before,
    start_ms,
    stop_ms,
    model,
    weight_decay,
    final_fft,
    add_bnorm,
    act_norm,
):
    model_name = model
    del model
    assert final_hz in [64, 256]

    car = not debug
    train_inputs, test_inputs = load_train_test(
        subject_id,
        car,
        n_sensors,
        final_hz,
        start_ms,
        stop_ms,
        half_before,
        only_load_given_sensors=debug,
    )

    cuda = True
    if cuda:
        train_inputs = [i.cuda() for i in train_inputs]
        test_inputs = [i.cuda() for i in test_inputs]

    from braindecode.datautil.signal_target import SignalAndTarget

    sets = []
    for inputs in (train_inputs, test_inputs):
        X = np.concatenate([var_to_np(ins) for ins in inputs]).astype(
            np.float32
        )
        y = np.concatenate(
            [np.ones(len(ins)) * i_class for i_class, ins in enumerate(inputs)]
        )
        y = y.astype(np.int64)
        set = SignalAndTarget(X, y)
        sets.append(set)
    train_set = sets[0]
    valid_set = sets[1]

    from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
    from braindecode.models.deep4 import Deep4Net
    from torch import nn
    from braindecode.torch_ext.util import set_random_seeds

    set_random_seeds(2019011641, cuda)
    n_chans = train_inputs[0].shape[1]
    n_time = train_inputs[0].shape[2]
    n_classes = 2
    input_time_length=train_set.X.shape[2]

    if model_name == 'shallow':
        # final_conv_length = auto ensures we only get a single output in the time dimension
        model = ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes,
                                input_time_length=input_time_length,
                                final_conv_length='auto')
    elif model_name == 'deep':
        model = Deep4Net(n_chans, n_classes,
                 input_time_length=train_set.X.shape[2],
                 pool_time_length=2,
                 pool_time_stride=2,
                 final_conv_length='auto')
    elif model_name == 'invertible':
        model = InvertibleModel(n_chans, n_time, final_fft=final_fft,
                                add_bnorm=add_bnorm)
    elif model_name == 'deep_invertible':
        n_chan_pad = 0
        filter_length_time = 11
        model = deep_invertible(
            n_chans, input_time_length,  n_chan_pad,  filter_length_time)
        model.add_module("select_dims", Expression(lambda x: x[:, :2, 0]))
        model.add_module("softmax", nn.LogSoftmax(dim=1))
        model = WrappedModel(model)

        ## set scale
        if act_norm:
            model.cuda()
            for module in model.network.modules():
                if hasattr(module, 'log_factor'):
                    module._forward_hooks.clear()
                    module.register_forward_hook(scale_to_unit_var)
            model.network(train_inputs[0].cuda());
            for module in model.network.modules():
                if hasattr(module, 'log_factor'):
                    module._forward_hooks.clear()

    else:
        assert False
    if cuda:
        model.cuda()

    from braindecode.torch_ext.optimizers import AdamW
    import torch.nn.functional as F
    if model_name == 'shallow':
        assert weight_decay == 'hardcoded'
        optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)
    elif model_name == 'deep':
        assert weight_decay == 'hardcoded'
        optimizer = AdamW(model.parameters(), lr=1 * 0.01,
                          weight_decay=0.5 * 0.001)  # these are good values for the deep model
    elif model_name == 'invertible':
        optimizer = AdamW(model.parameters(), lr=1e-4,
                          weight_decay=weight_decay)
    elif model_name == 'deep_invertible':
        optimizer = AdamW(model.parameters(), lr=1 * 0.001,
                          weight_decay=weight_decay)

    else:
        assert False

    model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, )
    model.fit(train_set.X, train_set.y, epochs=max_epochs, batch_size=64,
              scheduler='cosine',
              validation_data=(valid_set.X, valid_set.y), )

    return model.epochs_df, model.network


def save_torch_artifact(ex, obj, filename):
    """Uses tempfile and file lock to safely store a pkl object as artefact"""
    import tempfile
    import fasteners

    log.info("Saving torch artifact")
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmpfile:
        lockname = tmpfile.name + ".lock"
        file_lock = fasteners.InterProcessLock(lockname)
        file_lock.acquire()
        th.save(obj, open(tmpfile.name, "wb"))
        ex.add_artifact(tmpfile.name, filename)
        file_lock.release()
    log.info("Saved torch artifact")


def run(
    ex,
    debug,
    subject_id,
    max_epochs,
    n_sensors,
    final_hz,
    half_before,
    start_ms,
    stop_ms,
    model,
    save_model,
    weight_decay,
    only_return_exp,
    final_fft,
    add_bnorm,
    act_norm,
):
    kwargs = locals()
    kwargs.pop("ex")
    kwargs.pop("only_return_exp")
    kwargs.pop("save_model")
    th.backends.cudnn.benchmark = True
    import sys

    logging.basicConfig(
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )
    start_time = time.time()
    ex.info["finished"] = False
    confirm_gpu_availability()
    epochs_df, model = run_exp(**kwargs)

    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True
    last_row = epochs_df.iloc[-1]
    for key, val in last_row.iteritems():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
    save_pkl_artifact(ex, epochs_df, "epochs_df.pkl")
    if save_model:
        save_torch_artifact(ex, model, "model.pkl")

    print("Finished!")
