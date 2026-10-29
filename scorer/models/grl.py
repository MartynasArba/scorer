"""
Gradient reversal: identity in the forward pass, gradient * -alpha in the backward pass.
Lets the encoder learn features the domain classifier CANNOT separate (domain-adversarial training).
"""
import torch
from torch.autograd import Function


class GradientReversalLayer(Function):
    """Autograd function: identity forward, gradient multiplied by ``-alpha`` backward."""
    @staticmethod
    def forward(ctx, x, alpha):
        """Return ``x`` unchanged and remember ``alpha`` for the backward pass."""
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Return the reversed, scaled gradient for ``x`` and no gradient for ``alpha``."""
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal to ``x`` with strength ``alpha``."""
    return GradientReversalLayer.apply(x, alpha)
