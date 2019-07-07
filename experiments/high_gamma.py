import os
import site

os.sys.path.insert(0, "/home/schirrmr/code/reversible/")
os.sys.path.insert(0, "/home/schirrmr/braindecode/code/braindecode/")
import logging
import time
import numpy as np
import torch as th

from hyperoptim.parse import (
    cartesian_dict_of_lists_product,
    product_of_list_of_lists_of_dicts,
)
from hyperoptim.util import save_pkl_artifact
from braindecode.torch_ext.util import var_to_np, confirm_gpu_availability


from reversible2.distribution import TwoClassDist, TwoClassIndependentDist
from reversible2.monitor import compute_accs
from reversible2.models import smaller_model, larger_model, deep_invertible
from reversible2.training import CLFTrainer
from reversible2.classifier import SubspaceClassifier
from reversible2.monitor import compute_clf_accs
from reversible2.high_gamma import load_file, create_inputs, load_train_test


log = logging.getLogger(__name__)
log.setLevel("DEBUG")


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [
        {
            "save_folder": "/data/schirrmr/schirrmr/reversible/experiments/new-deep-invertible",
            "only_return_exp": False,
            "debug": False,
        }
    ]
    subject_id_params = dictlistprod({"subject_id": range(4,10)})
    data_params = dictlistprod({"n_sensors": [22], "final_hz": [256]})
    preproc_params = dictlistprod({"half_before": [True]})
    ival_params = [{"start_ms": 500, "stop_ms": 1500}]
    training_params = dictlistprod({"max_epochs": [1000, 4000]})

    model_params = dictlistprod({
        "model_name": ["deep_invertible"]
    })

    implementation_params = dictlistprod({"constant_memory": [True]})

    init_params = dictlistprod(
        {"data_zero_init": [False], "set_distribution_to_empirical": [True]}
    )

    network_params = dictlistprod({"final_fft": [False]})  # True

    clf_params = dictlistprod({"clf_loss": [None, 'sliced']})  # "likelihood", None

    dist_params = dictlistprod(
        {
            "ot_on_class_dims": [False],  # , True
            "independent_class_dists": [True],
        }
    )

    save_params = [{"save_model": True}]

    grid_params = product_of_list_of_lists_of_dicts(
        [
            default_params,
            subject_id_params,
            data_params,
            implementation_params,
            training_params,
            init_params,
            model_params,
            preproc_params,
            dist_params,
            ival_params,
            network_params,
            clf_params,
            save_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params


def run_exp(
    debug,
    subject_id,
    constant_memory,
    data_zero_init,
    set_distribution_to_empirical,
    ot_on_class_dims,
    max_epochs,
    independent_class_dists,
    n_sensors,
    clf_loss,
    final_hz,
    start_ms,
    stop_ms,
    half_before,
    final_fft,
    model_name,
):
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

    from reversible2.graph import Node
    from reversible2.branching import CatChans, ChunkChans, Select
    from reversible2.constantmemory import graph_to_constant_memory

    from copy import deepcopy
    from reversible2.graph import Node
    from reversible2.distribution import TwoClassDist
    from reversible2.wrap_invertible import WrapInvertible
    from reversible2.blocks import dense_add_no_switch, conv_add_3x3_no_switch
    from reversible2.rfft import RFFT, Interleave
    from reversible2.util import set_random_seeds
    import torch as th
    from reversible2.splitter import SubsampleSplitter

    set_random_seeds(2019011641, cuda)
    n_chans = train_inputs[0].shape[1]
    n_time = train_inputs[0].shape[2]
    if final_hz == 64:
        feature_model = smaller_model(
            n_chans, n_time, final_fft, constant_memory
        )
    else:
        assert final_hz == 256
        if model_name == 'old_invertible':
            feature_model = larger_model(
                n_chans, n_time, final_fft, constant_memory
            )
        elif model_name == 'deep_invertible':
            n_chan_pad = 0
            filter_length_time = 11
            feature_model = deep_invertible(
                n_chans, n_time,  n_chan_pad,  filter_length_time)
            from reversible2.view_as import ViewAs
            feature_model.add_module('flatten',
                                     ViewAs((-1, 176, 32), (-1, 5632)))
            from reversible2.graph import Node
            feature_model = Node(None, feature_model)
        else:
            assert False

    if cuda:
        feature_model.cuda()
    feature_model.eval()

    from reversible2.constantmemory import clear_ctx_dicts
    from reversible2.distribution import TwoClassDist

    if data_zero_init:
        feature_model.data_init(
            th.cat((train_inputs[0], train_inputs[1]), dim=0)
        )

    # Check that forward + inverse is really identical
    t_out = feature_model(train_inputs[0][:2])
    inverted = feature_model.invert(t_out)
    clear_ctx_dicts(feature_model)
    assert th.allclose(train_inputs[0][:2], inverted, rtol=1e-3, atol=1e-4)
    from reversible2.ot_exact import ot_euclidean_loss_for_samples

    if independent_class_dists:
        class_dist = TwoClassIndependentDist(
            np.prod(train_inputs[0].size()[1:])
        )
    else:
        class_dist = TwoClassDist(
            2, np.prod(train_inputs[0].size()[1:]) - 2, [0, 1]
        )
    class_dist.cuda()

    if set_distribution_to_empirical:
        for i_class in range(2):
            with th.no_grad():
                this_outs = feature_model(train_inputs[i_class])
                mean = th.mean(this_outs, dim=0)
                std = th.std(this_outs, dim=0)
                class_dist.set_mean_std(i_class, mean, std)
                # Just check
                setted_mean, setted_std = class_dist.get_mean_std(i_class)
                assert th.allclose(mean, setted_mean)
                assert th.allclose(std, setted_std)
        clear_ctx_dicts(feature_model)

    optim_model = th.optim.Adam(
        feature_model.parameters(), lr=1e-3, betas=(0.9, 0.999)
    )
    optim_dist = th.optim.Adam(
        class_dist.parameters(), lr=1e-2, betas=(0.9, 0.999)
    )

    if clf_loss is not None:
        clf = SubspaceClassifier(2, 10, np.prod(train_inputs[0].shape[1:]))
        clf.cuda()

        optim_clf = th.optim.Adam(clf.parameters(), lr=1e-3)
        clf_trainer = CLFTrainer(
            feature_model,
            clf,
            class_dist,
            optim_model,
            optim_clf,
            optim_dist,
            outs_loss=clf_loss,
        )

    import pandas as pd

    df = pd.DataFrame()

    from reversible2.training import OTTrainer

    trainer = OTTrainer(feature_model, class_dist, optim_model, optim_dist)

    from reversible2.constantmemory import clear_ctx_dicts
    from reversible2.timer import Timer

    i_start_epoch_out = int(np.round(max_epochs * 0.4)) + 1
    n_epochs = max_epochs + 1  # +1 for historical reasons.
    if debug:
        n_epochs = 21
        i_start_epoch_out = 5
    for i_epoch in range(n_epochs):
        epoch_row = {}
        with Timer(name="EpochLoop", verbose=False) as loop_time:
            loss_on_outs = i_epoch >= i_start_epoch_out
            result = trainer.train(
                train_inputs, loss_on_outs=(loss_on_outs and ot_on_class_dims)
            )
            if clf_loss is not None:
                result_clf = clf_trainer.train(
                    train_inputs, loss_on_outs=loss_on_outs
                )
                epoch_row.update(result_clf)

        epoch_row.update(result)
        epoch_row["runtime"] = loop_time.elapsed_secs * 1000
        acc_results = compute_accs(
            feature_model, train_inputs, test_inputs, class_dist
        )
        epoch_row.update(acc_results)
        if clf_loss is not None:
            clf_accs = compute_clf_accs(
                clf, feature_model, train_inputs, test_inputs
            )
            epoch_row.update(clf_accs)
        if i_epoch % (n_epochs // 20) != 0:
            df = df.append(epoch_row, ignore_index=True)
            # otherwise add ot loss in
        else:
            for i_class in range(len(train_inputs)):
                with th.no_grad():
                    class_ins = train_inputs[i_class]
                    samples = class_dist.get_samples(
                        i_class, len(train_inputs[i_class]) * 4
                    )
                    inverted = feature_model.invert(samples)
                    clear_ctx_dicts(feature_model)
                    ot_loss_in = ot_euclidean_loss_for_samples(
                        class_ins.view(class_ins.shape[0], -1),
                        inverted.view(inverted.shape[0], -1)[
                            : (len(class_ins))
                        ],
                    )
                    epoch_row[
                        "ot_loss_in_{:d}".format(i_class)
                    ] = ot_loss_in.item()
            df = df.append(epoch_row, ignore_index=True)
            print("Epoch {:d} of {:d}".format(i_epoch, n_epochs))
            print("Loop Time: {:.0f} ms".format(loop_time.elapsed_secs * 1000))

    return df, feature_model, class_dist


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
    constant_memory,
    data_zero_init,
    max_epochs,
    set_distribution_to_empirical,
    ot_on_class_dims,
    independent_class_dists,
    half_before,
    n_sensors,
    final_hz,
    start_ms,
    stop_ms,
    model_name,
    final_fft,
    clf_loss,
    save_model,
    only_return_exp,
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
    epochs_df, feature_model, class_dist = run_exp(**kwargs)

    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True
    last_row = epochs_df.iloc[-1]
    for key, val in last_row.iteritems():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
    save_pkl_artifact(ex, epochs_df, "epochs_df.pkl")
    if save_model:
        save_torch_artifact(ex, feature_model, "feature_model.pkl")
        save_torch_artifact(ex, class_dist, "class_dist.pkl")

    print("Finished!")
