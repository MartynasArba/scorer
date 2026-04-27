#pytorch autograd function: silent in forward pass, inversion during backward pass (* -alpha)
#this is so domain classifier learns the reverse thing: normally, a classifier separates functions, but here, it should merge them instead
#important so OOD datasets (like mine) are mapped to normal datasets
import torch
from torch.autograd import Function

class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha   
        return x.view_as(x) #nothing happens

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None #reversed gradient

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)