import torch as th
import matplotlib.pyplot as plt
import numpy as np
from reversible2.util import var_to_np
from reversible2.plot import display_close
from matplotlib.patches import Ellipse
import seaborn

def plot_outs(feature_model_a, train_inputs, test_inputs, class_dist):
     # Compute dist for mean/std of encodings
    data_cls_dists = []
    for i_class in range(len(train_inputs)):
        this_class_outs = feature_model_a(train_inputs[i_class])[:,:2]
        data_cls_dists.append(
            th.distributions.MultivariateNormal(th.mean(this_class_outs, dim=0),
            covariance_matrix=th.diag(th.std(this_class_outs, dim=0))))
    for setname, set_inputs in (("Train", train_inputs), ("Test", test_inputs)):

        outs = [feature_model_a(ins) for ins in set_inputs]
        c_outs = [o[:,:2] for o in outs]

        c_outs_all = th.cat(c_outs)

        cls_dists = []
        for i_class in range(len(c_outs)):
            mean, std = class_dist.get_mean_std(i_class)
            cls_dists.append(
                th.distributions.MultivariateNormal(mean[:2],covariance_matrix=th.diag(std[:2])))

        preds = th.stack([cls_dists[i_cls].log_prob(c_outs_all)
                         for i_cls in range(len(cls_dists))],
                        dim=-1)

        pred_labels = np.argmax(var_to_np(preds), axis=1)

        labels = np.concatenate([np.ones(len(set_inputs[i_cls])) * i_cls 
         for i_cls in range(len(train_inputs))])

        acc = np.mean(labels == pred_labels)
        
        data_preds = th.stack([data_cls_dists[i_cls].log_prob(c_outs_all)
                         for i_cls in range(len(cls_dists))],
                        dim=-1)
        data_pred_labels = np.argmax(var_to_np(data_preds), axis=1)
        data_acc = np.mean(labels == data_pred_labels)

        print("{:s} Accuracy: {:.2f}%".format(setname, acc * 100))
        fig = plt.figure(figsize=(5,5))
        ax = plt.gca()
        for i_class in range(len(c_outs)):
            o = var_to_np(c_outs[i_class]).squeeze()
            plt.scatter(o[:,0], o[:,1], s=20, alpha=0.75)
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
                       color=seaborn.color_palette()[i_class+2], s=80, marker="^")

        plt.title("{:6s} Accuracy:        {:.2f}%\n"
                  "From data mean/std: {:.2f}%".format(setname, acc * 100, data_acc * 100))
        plt.legend(("Right", "Rest", "Right Mean", "Rest Mean"))
        display_close(fig)