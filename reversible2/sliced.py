import torch as th


def sample_directions(n_dims, orthogonalize, cuda):
    if cuda:
        directions = th.cuda.FloatTensor(n_dims, n_dims).normal_(0, 1) + 1e-6
    else:
        directions = th.FloatTensor(n_dims, n_dims).normal_(0, 1) + 1e-6

    if orthogonalize:
        directions, r = th.qr(directions)
        d = th.diag(r, 0)
        ph = d.sign()
        directions *= ph
    directions = th.autograd.Variable(directions, requires_grad=False)
    norm_factors = th.norm(directions, p=2, dim=1, keepdim=True)
    directions = directions / norm_factors
    return directions


def norm_and_var_directions(directions):
    norm_factors = th.norm(directions, p=2, dim=1, keepdim=True)
    directions = directions / norm_factors
    return directions


def sliced_from_samples(samples_a, samples_b, n_dirs, adv_dirs,
                        orthogonalize=True, dist='w2'):
    assert (n_dirs > 0) or (adv_dirs is not None)
    dirs = [sample_directions(samples_a.size()[1], orthogonalize=orthogonalize,
                              cuda=samples_a.is_cuda) for _ in range(n_dirs)]
    if adv_dirs is not None:
        dirs = dirs + [adv_dirs]
    dirs = th.cat(dirs, dim=0)
    dirs = norm_and_var_directions(dirs)
    return sliced_from_samples_for_dirs(samples_a, samples_b, dirs, dist=dist)


def sliced_from_samples_for_dirs(samples_a, samples_b, dirs, dist):
    assert dist in ['w2', 'sqw2'] or callable(dist)

    if dirs is not None:
        projected_samples_a = th.mm(samples_a, dirs.t())
        projected_samples_b = th.mm(samples_b, dirs.t())
    else:
        projected_samples_a = samples_a
        projected_samples_b = samples_b
    sorted_samples_a, _ = th.sort(projected_samples_a, dim=0)
    sorted_samples_b, _ = th.sort(projected_samples_b, dim=0)
    n_a = len(sorted_samples_a)
    n_b = len(sorted_samples_b)
    if n_a > n_b:
        assert n_a % n_b == 0
        increase_factor = n_a // n_b
        sorted_samples_a = sorted_samples_a.view(n_a // increase_factor,
                                                 increase_factor,
                                                 sorted_samples_a.size()[1])
        sorted_samples_b = sorted_samples_b.unsqueeze(1)
    elif n_a < n_b:
        assert n_b % n_a == 0
        increase_factor = n_b // n_a
        sorted_samples_b = sorted_samples_b.view(n_b // increase_factor,
                                                 increase_factor,
                                                 sorted_samples_b.size()[1])
        sorted_samples_a = sorted_samples_a.unsqueeze(1)

    diffs = sorted_samples_a - sorted_samples_b
    # first sum across examples
    # (one W2-value per direction)
    # then mean across directions
    # then sqrt
    if callable(dist):
        loss = dist(diffs)
    elif dist == 'w2':
        eps = 1e-6
        loss = th.sqrt(th.mean(diffs * diffs) + eps)
    else:
        assert dist == 'sqw2'
        loss = th.mean(diffs * diffs)
    return loss