from functools import partial
from typing import Union

import torch.nn as nn
import torch.nn.init as init


def init_linear_(x: Union[nn.Linear, nn.Conv2d],
                 mode: str = 'trunc_',
                 std: float = .02,
                 fan: str = 'fan_in',
                 **kwargs):
    if not isinstance(x, (nn.Linear, nn.Conv2d)): return
    if mode in ('', 'trunc_'): kwargs['std'] = std
    if mode == 'kaiming_': kwargs['mode'] = fan
    getattr(init, f"{mode}normal_")(x.weight, **kwargs)
    if x.bias is not None: nn.init.zeros_(x.bias)

def init_linear(mode: str = 'trunc_', std: float = .02, fan: str = 'fan_in'):
    return partial(init_linear_, mode=mode, std=std, fan=fan)

def init_norm_(x: nn.Module, val: float = 1.0):
    if not isinstance(x, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)): return
    nn.init.constant_(x.weight, val)
    if x.bias is not None: nn.init.zeros_(x.bias)

def init_norm(val: float = 1.0):
    return partial(init_norm_, val=val)