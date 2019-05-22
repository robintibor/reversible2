import torch as th
from reversible2.gaussian import get_gauss_samples


class TwoClassDist(object):
    def __init__(self, n_class_dims, n_non_class_dims, truncate_to=3):
        super(TwoClassDist, self).__init__()
        self.class_means = th.zeros(n_class_dims, requires_grad=True)
        self.non_class_means = th.zeros(n_class_dims+n_non_class_dims, requires_grad=True)
        self.class_log_stds = th.zeros(n_class_dims, requires_grad=True)
        self.non_class_log_stds = th.zeros(n_class_dims+n_non_class_dims, requires_grad=True)
        self.truncate_to = truncate_to

    def get_mean_std(self, i_class):
        cur_mean = th.cat((self.non_class_means[:i_class],
                           self.class_means[i_class:i_class + 1],
                           self.non_class_means[i_class+1:]))
        cur_log_std = th.cat((self.non_class_log_stds[:i_class],
                              self.class_log_stds[i_class:i_class + 1],
                              self.non_class_log_stds[i_class+1:]))
        return cur_mean, th.exp(cur_log_std)

    def get_samples(self, i_class, n_samples):
        cur_mean, cur_std = self.get_mean_std(i_class)
        samples = get_gauss_samples(n_samples, cur_mean, cur_std, truncate_to=self.truncate_to)
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


    def change_to_other_class(self, outs, i_class_from, i_class_to, eps=1e-6):
        mean_from, std_from = self.get_mean_std(i_class_from)
        mean_to, std_to = self.get_mean_std(i_class_to)
        normed = (outs - mean_from.unsqueeze(0)) / (std_from.unsqueeze(0) + eps)
        transformed = (normed * std_to.unsqueeze(0)) + mean_to.unsqueeze(0)
        return transformed

    def get_class_log_prob(self, i_class, out,):
        mean, std = self.get_mean_std(i_class)
        cls_dist = th.distributions.MultivariateNormal(
                mean[:len(self.class_means)], covariance_matrix=
                th.diag(std[:len(self.class_log_stds)] ** 2))
        return cls_dist.log_prob(out[:,:len(self.class_means)])

    def get_total_log_prob(self, i_class, out, ):
        mean, std = self.get_mean_std(i_class)
        cls_dist = th.distributions.MultivariateNormal(
            mean, covariance_matrix=
            th.diag(std ** 2))
        return cls_dist.log_prob(out)

    def set_mean_std(self, i_class, mean, std):
        self.class_means.data[i_class] = mean.data[i_class]
        self.class_log_stds.data[i_class] = th.log(std).data[i_class]
        if i_class == 0:
            self.non_class_means.data[1:] = mean.data[1:]
            self.non_class_log_stds.data[1:] = th.log(std).data[1:]
        else:
            assert i_class == 1
            self.non_class_means.data[0] = mean.data[0]
            self.non_class_log_stds.data[0] = th.log(std).data[0]
            self.non_class_means.data[2:] = mean.data[2:]
            self.non_class_log_stds.data[2:] = th.log(std).data[2:]
