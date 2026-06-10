from typing import Optional, Union

from torch import nn
from creyone import cynn


def _consume_args(args: tuple, kwargs: dict) -> tuple[list, dict]:
    """Normalize positional args to (c1, c2, k), pulling missing values from kwargs.

    Accepts args in (c1, c2, k) order. Any leading args that are omitted are
    extracted from kwargs using the long-form keys ('in_channels', 'out_channels',
    'kernel_size') or their short aliases ('c1', 'c2', 'k').
    """
    args = list(args)
    if len(args) == 3: return args, kwargs
    k = kwargs.pop('kernel_size', kwargs.pop('k', None))    
    if len(args) == 2: return args + [k], kwargs
    c2 = kwargs.pop('out_channels', kwargs.pop('c2', None))
    if len(args) == 1: return args + [c2, k], kwargs
    c1 = kwargs.pop('in_channels', kwargs.pop('c1', None))
    return [c1, c2, k], kwargs


def _compute_same_padding(k: int, d: int = 1) -> Union[int, list[int]]:  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if isinstance(k, int): k = [k] 
    if d > 1: k = [d * (x - 1) + 1 for x in k]  # actual kernel-size
    p = [x // 2 for x in k]  # auto-pad
    return p[0] if len(p) == 1 else p


def _wrapfn(**kwargs):
    """Extract and return common convolution/pooling parameters from kwargs.

    Pops 'stride'/'s', 'padding'/'p', 'dilation'/'d', and 'groups'/'g' from
    kwargs, returning them as (s, p, d, g). The remaining kwargs are left for
    the caller to pass through to the underlying layer.
    """
    s = kwargs.pop('stride', kwargs.pop('s', 1))
    p = kwargs.pop('padding', kwargs.pop('p', 0))
    d = kwargs.pop('dilation', kwargs.pop('d', 1))
    g = kwargs.pop('groups', kwargs.pop('g', 1))
    return s, p, d, g


class AutoReshape(nn.Module):

    def __init__(self, mid_layer: nn.Module):
        super().__init__(); self._inner = mid_layer
    
    def forward(self, x: cynn.CreYonT) -> cynn.CreYonT:
        x = x.rearrange('B (H W) C -> B C H W', H=x.H).contiguous()
        return x(self._inner).rearrange('B C H W -> B (H W) C')


def wrap_conv(cls: nn.Conv2d, opt: Union[set, str] = set()):
    """Wrap a Conv Nd class with flexible argument parsing and optional behaviors.

    Args:
        cls: A ConvNd-compatible class to wrap.
        optional: A '+'-separated string of option flags:
            - 'grid': use the kernel size arg as the stride (grid-like sampling).
            - 'ap':   auto-pad so the output spatial size matches the input.
            - 'dw':   set groups = in_channels (depthwise convolution).

    Returns:
        A factory function that accepts (c1, c2, k, ...) positionally or as kwargs
        and forwards them to ``cls`` with stride, padding, dilation, and groups set.
    """
    if isinstance(opt, str): opt = set(opt.split('+'))

    def _fn(*args, **kwargs):
        args, kwargs = _consume_args(args, kwargs)
        s, p, d, g = _wrapfn(**kwargs)
        if 'grid' in opt: s = args[-1]
        if 'ap' in opt: p = _compute_same_padding(args[-1], d)
        if 'dw' in opt: g = args[0]

        ins = cls(*args, stride=s, padding=p, dilation=d, groups=g, **kwargs)
        if 'ar' in opt: ins = AutoReshape(ins)
        return ins
    
    return _fn


def wrap_pool(cls: nn.MaxPool2d, optional: str = None):
    """Wrap a pooling class with flexible argument parsing and optional behaviors.

    Args:
        cls: A PoolNd-compatible class to wrap.
        optional: A '+'-separated string of option flags:
            - 'grid': use the kernel size arg as the stride (grid-like sampling).
            - 'ap':   auto-pad so the output spatial size matches the input.

    Returns:
        A factory function that accepts (c1, c2, k, ...) positionally or as kwargs
        and forwards them to ``cls`` with stride, padding, and dilation set.
    """
    optional = (optional or '').split('+')

    def _fn(*args, **kwargs):
        args, kwargs = _consume_args(args, kwargs)
        s, p, d, _ = _wrapfn(**kwargs)
        if 'grid' in optional: s = args[-1]
        if 'ap' in optional: p = _compute_same_padding(args[-1], d)

        ins = cls(*args, stride=s, padding=p, dilation=d, **kwargs)
        if 'ar' in optional: ins = AutoReshape(ins)
        return ins
    
    return _fn



