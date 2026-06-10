import torch.nn as nn

from .utils.registry import register_layer


def _wrapper(cls, eps: float = 1e-5, **extra_kwargs):

    def _norm_func(*args, **kwargs):
        _eps = kwargs.pop('eps', eps)
        return cls(*args, eps=_eps, **extra_kwargs, **kwargs)

    return _norm_func


@register_layer('norm')
def layer(eps: float = 1e-5, **_):
    return _wrapper(nn.LayerNorm, eps=eps)

@register_layer('norm')
def batch(eps: float = 1e-5, dim: int = 2, mom: float = 0.1):
    if dim == 1: return _wrapper(nn.BatchNorm1d, eps=eps, momentum=mom)
    if dim == 2: return _wrapper(nn.BatchNorm2d, eps=eps, momentum=mom)
    raise ValueError(f"Unsupported dim={dim} for BatchNorm. Expected 1 or 2.")