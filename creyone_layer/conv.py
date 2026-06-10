from torch import nn

from .utils.registry import register_layer
from .wrap import wrap_conv


def _base(dim: int, opt: set | None = None):
    if opt is None: opt = set()
    if dim == 1: return wrap_conv(nn.Conv1d, opt=opt)
    if dim == 2: return wrap_conv(nn.Conv2d, opt=opt)
    if dim == 3: return wrap_conv(nn.Conv3d, opt=opt)
    raise ValueError(f'Unsupported dim: {dim}')


@register_layer('conv')
def base(dim: int = 2, optional: str = '') -> type[nn.Module]:
    return _base(dim, opt=set(optional.split('+')))


@register_layer('conv')
def depthwise(dim: int = 2, optional: str = '') -> type[nn.Module]:
    seq = set(optional.split('+')); seq.add('dw')
    return _base(dim, opt=seq)