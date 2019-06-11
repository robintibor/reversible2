import torch as th
import numpy as np
from reversible2.util import np_to_var
from reversible2.gradient_penalty import gradient_penalty
from reversible2.ot_exact import ot_euclidean_loss_for_samples
from reversible2.constantmemory import clear_ctx_dicts
from reversible2.ot_exact import ot_euclidean_loss_memory_saving_for_samples
from reversible2.gaussian import (
    transform_gaussians_by_dirs,
    get_gaussian_log_probs,
)
from reversible2.sliced import sliced_from_samples


class Trainer(object):
    def __init__(
        self,
        feature_model,
        adv_model,
        class_dist,
        optim_feature_model,
        optim_adv_model,
        optim_class_dist,
    ):
        self.feature_model = feature_model
        self.adv_model = adv_model
        self.class_dist = class_dist
        self.optim_feature_model = optim_feature_model
        self.optim_adv_model = optim_adv_model
        self.optim_class_dist = optim_class_dist

    def train(self, train_inputs, gen_update, loss_on_outs):
        result = dict()
        feature_model = self.feature_model
        adv_model = self.adv_model
        class_dist = self.class_dist
        optim_feature_model = self.optim_feature_model
        optim_adv_model = self.optim_adv_model
        optim_class_dist = self.optim_class_dist

        optim_feature_model.zero_grad()
        optim_class_dist.zero_grad()
        optim_adv_model.zero_grad()
        for i_class in range(len(train_inputs)):
            y = np_to_var([i_class]).cuda()
            class_ins = train_inputs[i_class]
            other_class_ins = train_inputs[1 - i_class]
            with th.set_grad_enabled(gen_update):
                samples = class_dist.get_samples(
                    i_class, len(train_inputs[i_class]) * 4
                )
                inverted = feature_model.invert(samples)
            with th.set_grad_enabled(gen_update):
                outs = feature_model(class_ins)
                changed_to_other_class = class_dist.change_to_other_class(
                    outs, i_class_from=i_class, i_class_to=1 - i_class
                )
                other_inverted = feature_model.invert(changed_to_other_class)
            if not gen_update:
                score_fake = adv_model(inverted.detach(), y)
                score_real = adv_model(class_ins, y)
                gradient_loss = gradient_penalty(
                    adv_model,
                    class_ins,
                    inverted[: (len(class_ins))].detach(),
                    y,
                )
                d_loss = (
                    -score_real.mean() + score_fake.mean() + gradient_loss * 100
                )
                d_loss.backward()

                score_real_other = adv_model(other_class_ins.detach(), 1 - y)
                score_fake_other = adv_model(other_inverted.detach(), 1 - y)
                gradient_loss_other = gradient_penalty(
                    adv_model,
                    other_class_ins,
                    other_inverted[: (len(class_ins))].detach(),
                    1 - y,
                )
                d_loss_other = (
                    -score_real_other.mean()
                    + score_fake_other.mean()
                    + gradient_loss_other * 100
                )
                d_loss_other.backward()

                # Clip gradient
                d_grad_norm = np.mean(
                    [
                        th.nn.utils.clip_grad_norm_([p], 100)
                        for p in adv_model.parameters()
                    ]
                )
                d_grad = np.mean(
                    [th.sum(p.grad ** 2).item() for p in adv_model.parameters()]
                )
                wd_d = -(-score_real.mean() + score_fake.mean()).item()
                result["wd_d_{:d}".format(i_class)] = wd_d
            else:
                ot_out_loss = ot_euclidean_loss_for_samples(
                    samples[:, i_class_inds],
                    outs[:, i_class_inds],
                )
                other_samples = class_dist.get_samples(
                    1 - i_class, len(train_inputs[1 - i_class]) * 4
                )
                ot_out_loss_other = ot_euclidean_loss_for_samples(
                    other_samples[:, i_class_inds],
                    changed_to_other_class[:, i_class_inds],
                )
                score_fake = adv_model(inverted, y)
                score_fake_other = adv_model(other_inverted, 1 - y)
                g_loss = -th.mean(score_fake) - th.mean(score_fake_other)
                if loss_on_outs:
                    g_loss = g_loss + ot_out_loss + ot_out_loss_other
                g_loss.backward()
                # Clip gradient
                g_grad_norm = np.mean(
                    [
                        th.nn.utils.clip_grad_norm_([p], 100)
                        for p in feature_model.parameters()
                    ]
                )
                g_grad = np.mean(
                    [
                        th.sum(p.grad ** 2).item()
                        for p in feature_model.parameters()
                    ]
                )
            clear_ctx_dicts(feature_model)
        if not gen_update:
            optim_adv_model.step()
        else:
            optim_feature_model.step()
            optim_class_dist.step()
        if gen_update:
            result.update(
                {
                    "g_loss": g_loss.item(),
                    "o_fake": th.mean(score_fake).item(),
                    "g_grad": g_grad,
                    "g_grad_norm": g_grad_norm,
                    "ot_out_loss": ot_out_loss.item(),
                    "ot_out_loss_other": ot_out_loss_other.item(),
                }
            )
        else:
            result.update(
                {
                    "d_loss": d_loss.item(),
                    "grad_loss": gradient_loss.item(),
                    "o_real": th.mean(score_real).item(),
                    "o_fake": th.mean(score_fake).item(),
                    "d_grad": d_grad,
                    "d_grad_norm": d_grad_norm,
                }
            )

        return result


class OTTrainer(object):
    def __init__(
        self, feature_model, class_dist, optim_feature_model, optim_class_dist
    ):
        self.feature_model = feature_model
        self.class_dist = class_dist
        self.optim_feature_model = optim_feature_model
        self.optim_class_dist = optim_class_dist

    def train(self, train_inputs, loss_on_outs):
        result = dict()
        feature_model = self.feature_model
        class_dist = self.class_dist
        optim_feature_model = self.optim_feature_model
        optim_class_dist = self.optim_class_dist

        optim_feature_model.zero_grad()
        optim_class_dist.zero_grad()
        n_min = min([len(t) for t in train_inputs])
        if hasattr(class_dist, 'i_class_inds'):
            i_class_inds = class_dist.i_class_inds
        else:
            i_class_inds = list(range(len(class_dist.get_mean_std(0)[0])))
        
        for i_class in range(len(train_inputs)):
            class_ins = train_inputs[i_class][:n_min]
            other_class_ins = train_inputs[1 - i_class][:n_min]
            samples = class_dist.get_samples(i_class, len(class_ins) * 4)
            inverted = feature_model.invert(samples)
            outs = feature_model(class_ins)
            changed_to_other_class = class_dist.change_to_other_class(
                outs, i_class_from=i_class, i_class_to=1 - i_class
            )
            other_inverted = feature_model.invert(changed_to_other_class)

            ot_in_loss = ot_euclidean_loss_memory_saving_for_samples(
                class_ins.view(len(class_ins), -1),
                inverted.view(len(inverted), -1),
            )
            ot_in_loss_other = ot_euclidean_loss_memory_saving_for_samples(
                other_class_ins.view(len(other_class_ins), -1),
                other_inverted.view(len(other_inverted), -1),
            )
            if loss_on_outs:
                ot_out_loss = ot_euclidean_loss_for_samples(
                    outs[:, i_class_inds],
                    samples[:, i_class_inds],
                )
            else:
                ot_out_loss = th.zeros(1)
            other_samples = class_dist.get_samples(
                1 - i_class, len(other_class_ins) * 4
            )
            if loss_on_outs:
                ot_out_loss_other = ot_euclidean_loss_for_samples(
                    changed_to_other_class[:, i_class_inds],
                    other_samples[:, i_class_inds],
                )
            else:
                ot_out_loss_other = th.zeros(1)

            g_loss = ot_in_loss + ot_in_loss_other
            if loss_on_outs:
                g_loss = g_loss + ot_out_loss + ot_out_loss_other
            g_loss.backward()
            # Clip gradient
            g_grad_norm = np.mean(
                [
                    th.nn.utils.clip_grad_norm_([p], 100)
                    for p in feature_model.parameters()
                ]
            )
            g_grad = np.mean(
                [th.sum(p.grad ** 2).item() for p in feature_model.parameters()]
            )
            result["ot_in_loss_{:d}".format(i_class)] = ot_in_loss.item()
            result[
                "ot_in_loss_other_{:d}".format(i_class)
            ] = ot_in_loss_other.item()
        clear_ctx_dicts(feature_model)
        optim_feature_model.step()
        optim_class_dist.step()
        result.update(
            {
                "g_loss": g_loss.item(),
                "g_grad": g_grad,
                "g_grad_norm": g_grad_norm,
                "ot_out_loss": ot_out_loss.item(),
                "ot_out_loss_other": ot_out_loss_other.item(),
            }
        )

        return result


class CLFTrainer(object):
    def __init__(
        self,
        feature_model,
        clf,
        dist,
        optim_model,
        optim_clf,
        optim_dist,
        outs_loss,
    ):
        assert outs_loss in ["sliced", "likelihood"]
        self.feature_model = feature_model
        self.clf = clf
        self.dist = dist
        self.optim_model = optim_model
        self.optim_clf = optim_clf
        self.optim_dist = optim_dist
        self.outs_loss = outs_loss

    def train(self, train_inputs, loss_on_outs):
        result = {}
        for i_class in range(2):
            outs = self.feature_model(train_inputs[i_class])
            soft_probs = self.clf(outs)
            clf_loss = th.nn.functional.nll_loss(
                soft_probs,
                th.ones(len(outs), device=soft_probs.device, dtype=th.int64)
                * i_class,
            )
            if self.outs_loss == "likelihood":
                m, s = self.dist.get_mean_std(i_class)
                transformed_means, transformed_stds = transform_gaussians_by_dirs(
                    m.unsqueeze(0), s.unsqueeze(0), self.clf.get_dirs().detach()
                )
                outs_projected = self.clf.project_outs(outs, detach_dirs=True)
                log_probs = get_gaussian_log_probs(
                    transformed_means[0],
                    th.log(transformed_stds[0]),
                    outs_projected,
                )
                subspace_loss = -th.mean(log_probs)
            else:
                assert self.outs_loss == "sliced"
                samples = self.dist.get_samples(i_class, len(outs) * 3)
                subspace_loss = sliced_from_samples(
                    outs,
                    samples,
                    n_dirs=0,
                    adv_dirs=self.clf.get_dirs().detach(),
                )
            loss = clf_loss + subspace_loss
            self.optim_dist.zero_grad()
            self.optim_model.zero_grad()
            self.optim_clf.zero_grad()
            loss.backward()
            self.optim_clf.step()
            if loss_on_outs:
                self.optim_model.step()
                self.optim_dist.step()

            result["clf_loss_{:d}".format(i_class)] = clf_loss.item()
            result["subspace_loss_{:d}".format(i_class)] = subspace_loss.item()
        return result
