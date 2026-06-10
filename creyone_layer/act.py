from functools import partial
from typing import Optional

import torch
from torch import nn

from .utils.registry import register_layer

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return SwishImplementation.apply(x)
    

class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

class HardSigmoid(nn.Module):

    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + 3) / 6
    
class HardSwish(nn.Module):

    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = HardSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
    

@register_layer('act')
def gelu(overwrite: Optional[nn.Module] = None, **kwargs):
    if overwrite is not None: return overwrite
    return nn.GELU

@register_layer('act')
def relu(overwrite: Optional[nn.Module] = None, inplace: bool = False, **kwargs):
    if overwrite is not None: return overwrite
    return partial(nn.ReLU, inplace=inplace)

@register_layer('act')
def relu6(overwrite: Optional[nn.Module] = None, **kwargs):
    if overwrite is not None: return overwrite
    return nn.ReLU6

@register_layer('act')
def sigmoid(overwrite: Optional[nn.Module] = None, **kwargs):
    if overwrite is not None: return overwrite
    return nn.Sigmoid

@register_layer('act')
def hardsig(overwrite: Optional[nn.Module] = None, inplace: bool = False, **kwargs):
    if overwrite is not None: return overwrite
    return partial(HardSigmoid, inplace=inplace)

@register_layer('act')
def hardswish(overwrite: Optional[nn.Module] = None, inplace: bool = False, **kwargs):
    if overwrite is not None: return overwrite
    return partial(HardSwish, inplace=inplace)

@register_layer('act')
def quickgelu(overwrite: Optional[nn.Module] = None, **kwargs):
    if overwrite is not None: return overwrite
    return QuickGELU

@register_layer('act')
def swisheff(overwrite: Optional[nn.Module] = None, **kwargs):
    if overwrite is not None: return overwrite
    return MemoryEfficientSwish