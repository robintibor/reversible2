import torch as th
from torch import nn
import numpy as np

from reversible2.sliced import norm_and_var_directions, sample_directions
from reversible2.gaussian import transform_gaussians_by_dirs
from reversible2.gaussian import get_gaussian_log_probs



class SubspaceClassifier(nn.Module):
    def __init__(self, n_classes, n_directions, n_dims):
        super(SubspaceClassifier, self).__init__()
        self.means = th.nn.Parameter(th.zeros(n_classes, n_directions))
        self.log_stds = th.nn.Parameter(th.zeros(n_classes, n_directions))
        classifier_dirs = sample_directions(
            np.prod(n_dims),
            orthogonalize=True, cuda=True)[:n_directions]
        self.classifier_dirs = th.nn.Parameter(classifier_dirs)

    def get_log_probs(self, outs):
        outs_for_clf = self.project_outs(outs, detach_dirs=False)
        return self.get_log_probs_projected(outs_for_clf)

    def log_softmax_from_projected(self, outs_for_clf):
        log_probs = self.get_log_probs_projected(outs_for_clf)
        return th.nn.functional.log_softmax(log_probs, dim=-1)

    def get_log_probs_projected(self, outs_for_clf):
        log_probs_per_class = []
        for i_class in range(len(self.means)):
            log_probs = get_gaussian_log_probs(self.means[i_class],
                                               self.log_stds[i_class],
                                               outs_for_clf)
            log_probs_per_class.append(log_probs)
        return th.stack(log_probs_per_class, dim=-1)

    def forward(self, outs):
        log_probs = self.get_log_probs(outs)
        return th.nn.functional.log_softmax(log_probs, dim=-1)

    def project_outs(self, outs, detach_dirs):
        normed_dirs = norm_and_var_directions(self.classifier_dirs)
        if detach_dirs:
            normed_dirs = normed_dirs.detach()
        projected_outs = th.mm(outs, normed_dirs.t(), )
        return projected_outs

    def get_dirs(self):
        normed_dirs = norm_and_var_directions(self.classifier_dirs)
        return normed_dirs



