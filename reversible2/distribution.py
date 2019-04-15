import torch as th
from reversible2.gaussian import get_gauss_samples


class TwoClassDist(object):
    def __init__(self, ):
        super(TwoClassDist, self).__init__()
        self.class_means = th.zeros(2, requires_grad=True)
        self.non_class_means = th.zeros(62, requires_grad=True)
        self.class_log_stds = th.zeros(2, requires_grad=True)
        self.non_class_log_stds = th.zeros(62, requires_grad=True)

    def get_mean_std(self, i_class):
        device = self.class_means.device
        cur_mean = th.cat((th.zeros(i_class, device=device),
                           self.class_means[i_class:i_class + 1],
                           th.zeros(len(self.class_means) - i_class - 1,
                                    device=device),
                           self.non_class_means))
        cur_log_std = th.cat((th.ones(i_class, device=device) * -9,
                              self.class_log_stds[i_class:i_class + 1],
                              th.ones(len(self.class_log_stds) - i_class - 1,
                                      device=device) * -9,
                              self.non_class_log_stds))
        return cur_mean, th.exp(cur_log_std)

    def get_samples(self, i_class, n_samples):
        cur_mean, cur_std = self.get_mean_std(i_class)
        samples = get_gauss_samples(n_samples, cur_mean, cur_std)
        return samples

    def cuda(self):
        self.class_means.data = self.class_means.data.cuda()
        self.non_class_means.data = self.non_class_means.data.cuda()
        self.class_log_stds.data = self.class_log_stds.data.cuda()
        self.non_class_log_stds.data = self.non_class_log_stds.data.cuda()
        return self

    def parameters(self):
        return [self.class_means, self.non_class_means, self.class_log_stds,
                self.non_class_log_stds]