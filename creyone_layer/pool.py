from torch import nn

from .utils.registry import register_layer
from .wrap import wrap_pool


def _max(dim: int, opt: set | None = None):
    if opt is None: opt = set()
    return wrap_pool(getattr(nn, f"MaxPool{dim}d"), opt=opt)


def _avg(dim: int, opt: set | None = None):
    if opt is None: opt = set()
    return wrap_pool(getattr(nn, f"AvgPool{dim}d"), opt=opt)


@register_layer('pool')
def max(dim: int = 2, optional: str = '') -> callable:
    return _max(dim, opt=set(optional.split('+')))


@register_layer('pool')
def avg(dim: int = 2, optional: str = '') -> callable:
    return _avg(dim, opt=set(optional.split('+')))


@register_layer('pool')
def aavg(dim: int = 2, optional: str = '') -> callable:
    return getattr(nn, f"AdaptiveAvgPool{dim}d")
