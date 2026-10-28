"""
Gradient reversal: identity in the forward pass, gradient * -alpha in the backward pass.
Lets the encoder learn features the domain classifier CANNOT separate (domain-adversarial training).
"""
import torch
from torch.autograd import Function


class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversalLayer.apply(x, alpha)
