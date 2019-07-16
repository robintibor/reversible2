import torch as th
import matplotlib.pyplot as plt
import numpy as np
from reversible2.util import var_to_np
from reversible2.plot import display_close
from matplotlib.patches import Ellipse
import seaborn

def plot_outs(feature_model, train_inputs, test_inputs, class_dist):
    with th.no_grad():
        # Compute dist for mean/std of encodings
        data_cls_dists = []
        for i_class in range(len(train_inputs)):
            this_class_outs = feature_model(train_inputs[i_class])[:,:2]
            data_cls_dists.append(
                th.distributions.MultivariateNormal(th.mean(this_class_outs, dim=0),
                covariance_matrix=th.diag(th.std(this_class_outs, dim=0) ** 2)))
        for setname, set_inputs in (("Train", train_inputs), ("Test", test_inputs)):

            outs = [feature_model(ins) for ins in set_inputs]
            c_outs = [o[:,:2] for o in outs]

            c_outs_all = th.cat(c_outs)

            cls_dists = []
            for i_class in range(len(c_outs)):
                mean, std = class_dist.get_mean_std(i_class)
                cls_dists.append(
                    th.distributions.MultivariateNormal(mean[:2],covariance_matrix=th.diag(std[:2] ** 2)))

            preds_per_class = [th.stack([cls_dists[i_cls].log_prob(c_out)
                             for i_cls in range(len(cls_dists))],
                            dim=-1) for c_out in c_outs]

            pred_labels_per_class = [np.argmax(var_to_np(preds), axis=1)
                           for preds in preds_per_class]

            labels = np.concatenate([np.ones(len(set_inputs[i_cls])) * i_cls 
             for i_cls in range(len(train_inputs))])

            acc = np.mean(labels == np.concatenate(pred_labels_per_class))

            data_preds_per_class = [th.stack([data_cls_dists[i_cls].log_prob(c_out)
                             for i_cls in range(len(cls_dists))],
                            dim=-1) for c_out in c_outs]
            data_pred_labels_per_class = [np.argmax(var_to_np(data_preds), axis=1)
                                for data_preds in data_preds_per_class]
            data_acc = np.mean(labels == np.concatenate(data_pred_labels_per_class))

            print("{:s} Accuracy: {:.1f}%".format(setname, acc * 100))
            fig = plt.figure(figsize=(5,5))
            ax = plt.gca()
            for i_class in range(len(c_outs)):
                #if i_class == 0:
                #    continue
                o = var_to_np(c_outs[i_class]).squeeze()
                incorrect_pred_mask = pred_labels_per_class[i_class] != i_class
                plt.scatter(o[:,0], o[:,1], s=20, alpha=0.75, label=["Right", "Rest"][i_class])
                assert len(incorrect_pred_mask) == len(o)
                plt.scatter(o[incorrect_pred_mask,0], o[incorrect_pred_mask,1], marker='x', color='black',
                           alpha=1, s=5)
                means, stds = class_dist.get_mean_std(i_class)
                means = var_to_np(means)[:2]
                stds = var_to_np(stds)[:2]
                for sigma in [0.5,1,2,3]:
                    ellipse = Ellipse(means, stds[0]*sigma, stds[1]*sigma)
                    ax.add_artist(ellipse)
                    ellipse.set_edgecolor(seaborn.color_palette()[i_class])
                    ellipse.set_facecolor("None")
            for i_class in range(len(c_outs)):
                o = var_to_np(c_outs[i_class]).squeeze()
                plt.scatter(np.mean(o[:,0]), np.mean(o[:,1]),
                           color=seaborn.color_palette()[i_class+2], s=80, marker="^",
                           label=["Right Mean", "Rest Mean"][i_class])

            plt.title("{:6s} Accuracy:        {:.1f}%\n"
                      "From data mean/std: {:.1f}%".format(setname, acc * 100, data_acc * 100))
            plt.legend(bbox_to_anchor=(1,1,0,0))
            display_close(fig)
    return