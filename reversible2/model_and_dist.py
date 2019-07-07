import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np
from reversible2.constantmemory import clear_ctx_dicts
from reversible2.invert import invert
from reversible2.distribution import TwoClassIndependentDist


class ModelAndDist(nn.Module):
    def __init__(self, model, dist):
        super(ModelAndDist, self).__init__()
        self.model = model
        self.dist = dist

    def get_examples(self, i_class, n_samples,):
        samples = self.dist.get_samples(i_class=i_class, n_samples=n_samples)
        if hasattr(self.model, 'invert'):
            examples = self.model.invert(samples)
        else:
            examples = invert(self.model, samples)
        return examples

    def get_total_log_prob(self, i_class, class_ins):
        outs = self.model(class_ins)
        return self.dist.get_total_log_prob(i_class, outs)


    def log_softmax(self, inputs):
        log_probs = th.stack([self.get_total_log_prob(i_class, inputs)
                              for i_class in range(2)], dim=-1)
        log_softmaxed = F.log_softmax(log_probs, dim=1)
        return log_softmaxed

    def set_dist_to_empirical(self, inputs):
        set_dist_to_empirical(self.model, self.dist, inputs)


def set_dist_to_empirical(feature_model, class_dist, inputs, min_std=None):
    for i_class in range(len(inputs)):
        with th.no_grad():
            if class_dist.get_mean_std(i_class)[0].is_cuda:
                class_ins = inputs[i_class].cuda()
            else:
                class_ins = inputs[i_class]
            this_outs = feature_model(class_ins)
            mean = th.mean(this_outs, dim=0)
            std = th.std(this_outs, dim=0)
            if min_std is not None:
                std.data.clamp_min_(min_std)
            class_dist.set_mean_std(i_class, mean, std)
            # Just check
            setted_mean, setted_std = class_dist.get_mean_std(i_class)
            assert th.allclose(mean, setted_mean)
            assert th.allclose(std, setted_std)
    clear_ctx_dicts(feature_model)


def create_empirical_dist(model, inputs, min_std=None):
    data_dist = TwoClassIndependentDist(np.prod(inputs[0].size()[1:]))
    data_dist.cuda()
    set_dist_to_empirical(model, data_dist, inputs, min_std=min_std)
    return data_dist