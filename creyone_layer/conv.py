from typing import Optional

import torch
from torch import nn

from .utils.registry import register_layer
from .wrap import wrap_conv


def _base(dim: int, opt: str = None):
    if dim == 1: return wrap_conv(nn.Conv1d, optional=opt)
    if dim == 2: return wrap_conv(nn.Conv2d, optional=opt)
    if dim == 3: return wrap_conv(nn.Conv3d, optional=opt)


@register_layer('conv')
def base(dim: 2, optional: str = '') -> type[nn.Module]:
    return _base(dim, opt=optional)


@register_layer('conv')
def depthwise(dim: 2, optional: str = '') -> type[nn.Module]:
    seq = optional.split('+'); seq.append('dw')
    return _base(dim, opt=seq)