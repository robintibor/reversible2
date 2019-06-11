import os
import site
os.sys.path.insert(0, "/home/schirrmr/code/reversible/")
os.sys.path.insert(0, "/home/schirrmr/braindecode/code/braindecode/")
import torch as th
import numpy as np
from reversible2.util import var_to_np, np_to_var


def compute_accs(feature_model, train_inputs, test_inputs, class_dist):
    with th.no_grad():
        # Compute dist for mean/std of encodings
        data_cls_dists = []
        if hasattr(class_dist, 'i_class_inds'):
            i_class_inds = class_dist.i_class_inds
        else:
            i_class_inds = list(range(len(class_dist.get_mean_std(0)[0])))
        for i_class in range(len(train_inputs)):
            this_class_outs = feature_model(train_inputs[i_class])[
                :, i_class_inds
            ]
            data_cls_dists.append(
                th.distributions.MultivariateNormal(
                    th.mean(this_class_outs, dim=0),
                    covariance_matrix=th.diag(th.std(this_class_outs, dim=0) ** 2),
                )
            )
        results = {}
        for setname, set_inputs in (("Train", train_inputs), ("Test", test_inputs)):
            outs = [feature_model(ins) for ins in set_inputs]
            c_outs = [o[:, i_class_inds] for o in outs]

            cls_dists = []
            for i_class in range(len(c_outs)):
                mean, std = class_dist.get_mean_std(i_class)
                cls_dists.append(
                    th.distributions.MultivariateNormal(
                        mean[i_class_inds],
                        covariance_matrix=th.diag(std[i_class_inds] ** 2),
                    )
                )

            preds_per_class = [
                th.stack(
                    [
                        cls_dists[i_cls].log_prob(c_out)
                        for i_cls in range(len(cls_dists))
                    ],
                    dim=-1,
                )
                for c_out in c_outs
            ]

            pred_labels_per_class = [
                np.argmax(var_to_np(preds), axis=1) for preds in preds_per_class
            ]

            labels = np.concatenate(
                [
                    np.ones(len(set_inputs[i_cls])) * i_cls
                    for i_cls in range(len(train_inputs))
                ]
            )

            acc = np.mean(labels == np.concatenate(pred_labels_per_class))

            data_preds_per_class = [
                th.stack(
                    [
                        data_cls_dists[i_cls].log_prob(c_out)
                        for i_cls in range(len(cls_dists))
                    ],
                    dim=-1,
                )
                for c_out in c_outs
            ]
            data_pred_labels_per_class = [
                np.argmax(var_to_np(data_preds), axis=1)
                for data_preds in data_preds_per_class
            ]
            data_acc = np.mean(labels == np.concatenate(data_pred_labels_per_class))
            results["{:s}_acc".format(setname.lower())] = acc
            results["{:s}_data_acc".format(setname.lower())] = data_acc
    return results


def compute_clf_accs(clf, feature_model, train_inputs, test_inputs,):
    results = {}
    for setname, set_inputs in (("Train", train_inputs), ("Test", test_inputs)):
        outs = [feature_model(ins) for ins in set_inputs]
        preds_per_class = [clf(o) for o in outs]

        pred_labels_per_class = [np.argmax(var_to_np(preds), axis=1)
                       for preds in preds_per_class]

        labels = np.concatenate([np.ones(len(set_inputs[i_cls])) * i_cls
         for i_cls in range(len(train_inputs))])

        acc = np.mean(labels == np.concatenate(pred_labels_per_class))
        results['{:s}_clf_acc'.format(setname.lower())] = acc
    return results