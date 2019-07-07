import numpy as np
from torch import nn
import torch as th
import torch.nn.functional as F

from reversible2.gaussian import get_gauss_samples
from reversible2.util import flatten_2d


class GaussianMixture(nn.Module):
    def __init__(self, means, log_stds):
        super(GaussianMixture, self).__init__()
        self.means = means
        self.log_stds = log_stds

    def log_probs(self, points, ):
        # demeaned will be points x mixture components x dims
        demeaned = points.unsqueeze(1) - self.means.unsqueeze(0)
        # still points x mixture components x dims
        rescaled = demeaned / th.exp(self.log_stds).unsqueeze(0)

        log_probs_per_dim = -((rescaled ** 2) / 2) - np.log(
            float(np.sqrt(2 * np.pi))) - self.log_stds.unsqueeze(0)
        # sum over dimensions
        log_probs_per_mixture = th.sum(log_probs_per_dim, dim=-1)

        log_probs_per_point = th.logsumexp(log_probs_per_mixture,
                                           dim=1) - np.log(
            log_probs_per_mixture.shape[1])
        return log_probs_per_point

    def sample(self, n_samples):
        n_samples_per_mixture = np.bincount(
            np.random.choice(len(self.means), size=n_samples))
        samples = []
        for i_mixture, n in enumerate(n_samples_per_mixture):
            if n > 0:
                samples.append(
                    get_gauss_samples(
                        n,
                        self.means[i_mixture],
                        th.exp(self.log_stds[i_mixture]))
                )
        return th.cat(samples, dim=0)



class TwoClassMixture(nn.Module):
    def __init__(self, mixtures):
        super(TwoClassMixture, self).__init__()
        self.mixtures = mixtures

    def log_softmax(self, points):
        log_probs = th.stack([m.log_probs(points) for m in self.mixtures],
                             dim=-1)
        return F.log_softmax(log_probs, dim=-1)
