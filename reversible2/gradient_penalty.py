from torch import autograd
import torch as th
from torch import nn

def gradient_penalty(critic, real_data, generated_data, y=None):
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

    # Calculate score of interpolated examples
    if y is None:
        score_interpolated = critic(interpolated)
    else:
        score_interpolated = critic(interpolated, y)

    # Calculate gradients of scores with respect to examples
    gradients = autograd.grad(outputs=score_interpolated, inputs=interpolated,
                           grad_outputs=th.ones(score_interpolated.size()).cuda()
                           if real_data.is_cuda else th.ones(
                               score_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = th.sqrt(th.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return (nn.functional.relu(gradients_norm - 1) ** 2).mean()
