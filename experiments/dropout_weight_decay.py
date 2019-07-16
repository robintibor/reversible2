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
from reversible2.monitor import compute_clf_accs
from reversible2.high_gamma import load_train_test
from reversible2.model_and_dist import ModelAndDist
from reversible2.util import  flatten_2d
from reversible2.ot_exact import  get_matched_samples
import torch.nn.functional as F

log = logging.getLogger(__name__)
log.setLevel("DEBUG")


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [
        {
            "save_folder": "/data/schirrmr/schirrmr/reversible/experiments/dropout-weight-decay",
            "only_return_exp": False,
            "debug": False,
        }
    ]
    subject_id_params = dictlistprod({"subject_id": range(4,5)})
    data_params = dictlistprod({"n_sensors": [22], "final_hz": [256]})
    preproc_params = dictlistprod({"half_before": [True]})
    ival_params = [{"start_ms": 500, "stop_ms": 1500}]
    training_params = dictlistprod({"max_epochs": [600,1200]})

    implementation_params = dictlistprod({"constant_memory": [False]})

    init_params = dictlistprod(
        {"set_distribution_to_empirical": [True]}
    )

    optim_params =  dictlistprod(
        {
            "uni_noise_factor": [0, 1e-2, 5e-2,],#
            "gauss_noise_factor": [0.1],  #
            "weight_decay": [1e-4,], #
         }
    )


    network_params = dictlistprod({"final_fft": [True, False],
                                   'drop_p': [0,0.1,]})  #


    save_params = [{"save_model": True}]


    grid_params = product_of_list_of_lists_of_dicts(
        [
            default_params,
            subject_id_params,
            data_params,
            implementation_params,
            training_params,
            init_params,
            preproc_params,
            ival_params,
            optim_params,
            network_params,
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
    set_distribution_to_empirical,
    max_epochs,
    n_sensors,
    final_hz,
    start_ms,
    stop_ms,
    half_before,
    drop_p,
    uni_noise_factor,
    gauss_noise_factor,
    weight_decay,
    final_fft,
):
    assert final_hz == 256

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

    from reversible2.util import set_random_seeds
    import torch as th

    set_random_seeds(2019011641, cuda)
    n_chans = train_inputs[0].shape[1]
    n_time = train_inputs[0].shape[2]
    model = larger_model(
        n_chans, n_time, final_fft, kernel_length=11, constant_memory=constant_memory
    )

    if cuda:
        model.cuda()
    model.eval()

    if drop_p > 0:
        from reversible2.models import add_dropout_before_convs
        add_dropout_before_convs(model, p_conv=drop_p, p_full=drop_p)



    from reversible2.constantmemory import clear_ctx_dicts
    model.eval()
    # Check that forward + inverse is really identical
    t_out = model(train_inputs[0][:2])
    inverted = model.invert(t_out)
    clear_ctx_dicts(model)
    assert th.allclose(train_inputs[0][:2], inverted, rtol=1e-3, atol=1e-4)
    dist = TwoClassIndependentDist(np.prod(train_inputs[0].size()[1:]))
    dist.cuda()
    model_and_dist = ModelAndDist(model, dist)

    if set_distribution_to_empirical:
        with th.no_grad():
            model_and_dist.set_dist_to_empirical(train_inputs)
        clear_ctx_dicts(model)

    optim = th.optim.Adam([{'params': dist.parameters(), 'lr': 1e-2},
                           {'params': list(model_and_dist.model.parameters()),
                            'lr': 1e-4,
                            'weight_decay': weight_decay}])


    import pandas as pd
    from reversible2.model_and_dist import create_empirical_dist

    df = pd.DataFrame()

    from reversible2.constantmemory import clear_ctx_dicts
    from reversible2.timer import Timer

    n_epochs = max_epochs + 1  # +1 for historical reasons.
    if debug:
        n_epochs = 21

    for i_epoch in range(n_epochs):
        with Timer(verbose=False) as timer:
            if i_epoch > 0:
                model_and_dist.model.train()
                optim.zero_grad()
                for i_class in range(2):
                    class_ins = train_inputs[i_class].cuda()
                    noise = (th.rand_like(
                            class_ins) - 0.5) * uni_noise_factor + (
                        th.randn_like(class_ins) * gauss_noise_factor
                    )
                    log_probs = model_and_dist.get_total_log_prob(
                        i_class, class_ins + noise)
                    loss = -th.mean(log_probs)
                    loss.backward()

                optim.step()
                model.eval()
                clear_ctx_dicts(model_and_dist.model)
        # only record stuff every 10th epoch
        if i_epoch % 10 == 0:
            print("Epoch {:d} of {:d}".format(i_epoch, n_epochs))
            print("Loop Time: {:.0f} ms".format(timer.elapsed_secs * 1000))

            model.eval()
            with th.no_grad():
                lip_losses = []
                for i_class in range(2):
                    samples = model_and_dist.dist.get_samples(i_class, 25).detach()
                    in_samples = model_and_dist.model.invert(samples)

                    out_diffs = th.norm(samples.unsqueeze(0) - samples.unsqueeze(1),
                                        p=2, dim=2)

                    in_diffs = th.norm(
                        flatten_2d(in_samples).unsqueeze(0) - flatten_2d(
                            in_samples).unsqueeze(1), p=2, dim=2)

                    out_diffs = out_diffs.flatten()
                    in_diffs = in_diffs.flatten()

                    assert len(out_diffs.shape) == 1
                    assert len(in_diffs.shape) == 1
                    ratio = in_diffs / th.clamp(out_diffs, min=1e-9)
                    assert len(ratio.shape) == 1
                    lip_loss = th.mean(F.relu(ratio) ** 2)
                    lip_losses.append(lip_loss.item())

                epoch_row = {'lip_loss': np.mean(lip_losses)}

                for setname, inputs in (
                    ("train", train_inputs), ("test", test_inputs),):
                    OTs = []
                    nlls = []
                    inputs = [i.cuda() for i in inputs]
                    for i_class in range(2):
                        examples = model_and_dist.get_examples(i_class, len(
                            inputs[i_class]) * 4)
                        matched_examples = get_matched_samples(
                            flatten_2d(inputs[i_class]), flatten_2d(examples))
                        OT = th.mean(th.norm(
                            flatten_2d(inputs[i_class]).unsqueeze(
                                1) - matched_examples, p=2, dim=2))  #
                        nll = -th.mean(
                            model_and_dist.get_total_log_prob(i_class,
                                                              inputs[i_class]))
                        OTs.append(OT.item())
                        nlls.append(nll.item())
                    epoch_row[setname + '_OT'] = np.mean(OTs)
                    epoch_row[setname + '_NLL'] = np.mean(nlls)

                for setname, inputs in (
                    ("train", train_inputs), ("test", test_inputs)):
                    corrects = []
                    inputs = [i.cuda() for i in inputs]
                    for i_class in range(2):
                        outs = model_and_dist.log_softmax(
                            inputs[i_class].cuda())
                        pred_label = np.argmax(var_to_np(outs), axis=1)
                        correct = pred_label == i_class
                        corrects.append(correct)
                    acc = np.mean(np.concatenate(corrects))
                    epoch_row['model_' + setname + '_acc'] = acc

                for name, inputs in (("train", train_inputs),
                                     ("combined",
                                      [th.cat((train_inputs[i_class].cuda(),
                                               test_inputs[i_class].cuda()), dim=0)
                                       for i_class in range(2)]),
                                     ("test", test_inputs)):
                    emp_dist = create_empirical_dist(model_and_dist.model, inputs)

                    emp_model_dist = ModelAndDist(model_and_dist.model, emp_dist)
                    for setname, inner_inputs in (
                        ("train", train_inputs), ("test", test_inputs)):
                            inner_inputs = [i.cuda() for i in inner_inputs]
                            corrects = []
                            for i_class in range(2):
                                outs = emp_model_dist.log_softmax(
                                    inner_inputs[i_class].cuda())
                                pred_label = np.argmax(var_to_np(outs), axis=1)
                                correct = pred_label == i_class
                                corrects.append(correct)
                            acc = np.mean(np.concatenate(corrects))
                            epoch_row[name + '_' + setname + '_acc'] = acc
            epoch_row['traintime'] = timer.elapsed_secs * 1000
            df = df.append(epoch_row, ignore_index=True)

            print(df.iloc[-1])

    return df, model_and_dist


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
    max_epochs,
    set_distribution_to_empirical,
    half_before,
    n_sensors,
    final_hz,
    start_ms,
    stop_ms,
    final_fft,
    drop_p,
    uni_noise_factor,
    gauss_noise_factor,
    weight_decay,
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
    epochs_df, model_and_dist = run_exp(**kwargs)

    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True
    last_row = epochs_df.iloc[-1]
    for key, val in last_row.iteritems():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time
    save_pkl_artifact(ex, epochs_df, "epochs_df.pkl")
    if save_model:
        save_torch_artifact(ex, model_and_dist, "model_and_dist.pkl")

    print("Finished!")
