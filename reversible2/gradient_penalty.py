from torch import autograd
import torch as th
from torch import nn


def gradient_penalty(critic, real_data, generated_data, y=None, max_grad=1):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha_dims = (batch_size,) + (1,) * (len(real_data.shape) - 1)
    alpha = th.rand(alpha_dims)
    alpha = alpha.expand_as(real_data)
    if real_data.is_cuda:
        alpha = alpha.cuda()
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = interpolated.detach().requires_grad_(True)
    if real_data.is_cuda:
        interpolated = interpolated.cuda()
    return gradient_penalty_at_points(
        critic=critic, points=interpolated, y=y, max_grad=max_grad
    )


def gradient_penalty_at_points(critic, points, y=None, max_grad=1):
    if y is None:
        score = critic(points)
    else:
        score = critic(points, y)

    # Calculate gradients of scores with respect to examples
    gradients = autograd.grad(
        outputs=score,
        inputs=points,
        grad_outputs=th.ones(score.size(), device=score.device),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon

    gradients_norm = th.norm(
        gradients, p=2, dim=tuple(range(gradients.ndimension()))[1:]
    )
    # Return gradient penalty
    return (nn.functional.relu(gradients_norm - max_grad) ** 2).mean()
