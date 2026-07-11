from functools import partial
from typing import Union

import torch
import torch.nn as nn
import torch.nn.init as init

from .linear import LoRALinear


def same_as_linear(w: torch.Tensor, group_dim: int = None):
    """Initialize a weight tensor the same way ``nn.Linear`` initializes its weight.

    Args:
        w: Weight tensor to initialize in-place. Must be 2D, unless ``group_dim``
            is given, in which case ``w`` must be 2D after removing ``group_dim``
            (i.e. every slice along ``group_dim`` is 2D).
        group_dim: If given, apply the initialization independently to each
            slice along this dimension instead of to ``w`` as a whole.
    """
    if group_dim is None:
        if w.dim() != 2:
            raise ValueError("w must be 2D unless group_dim is given")
        nn.init.kaiming_uniform_(w, a = 5 ** 0.5)
        return
    if w.dim() != 3:
        raise ValueError("w must be 2D after removing group_dim")
    for i in range(w.shape[group_dim]):
        nn.init.kaiming_uniform_(w.select(group_dim, i), a = 5 ** 0.5)


def init_lora_(x: LoRALinear, mode: str = 'trunc_', 
               general_init: bool = True, zeroB: bool = True, 
               linear_init: bool = False, **kwargs):
    """Initialize LoRA adapter weights in-place.

    Initializes adwA with Kaiming uniform (general) or truncated/normal distribution,
    and adwB with zeros (general) or the specified distribution.

    Args:
        x: A LoRALinear module whose parameters will be initialized.
        mode: Prefix for the ``torch.nn.init`` function to use when
            ``general_init=False`` (e.g. ``'trunc_'`` -> ``trunc_normal_``).
        general_init: If True, use Kaiming uniform for adwA and zeros for adwB.
            If False, apply the distribution specified by ``mode`` to both.
        zeroB: If True (and ``general_init`` is False), still initialize adwB
            with zeros instead of the distribution specified by ``mode``.
        linear_init: If True, also initialize the base linear weights of ``x``
            via ``init_linear_`` before initializing the LoRA adapters.
        **kwargs: Additional keyword arguments forwarded to the init function.
    """
    if linear_init: init_linear_(x, mode=mode, **kwargs)
    for k, v in x.named_parameters():
        if k.split('.')[-1] == 'adwA':
            if general_init:
                same_as_linear(v)
            else:
                getattr(init, f"{mode}normal_")(v, **kwargs)
        if k.split('.')[-1] == 'adwB':
            if general_init or zeroB:
                nn.init.zeros_(v)
            else:
                getattr(init, f"{mode}normal_")(v, **kwargs)


def init_linear_(x: Union[nn.Linear, nn.Conv2d],
                 mode: str = 'trunc_',
                 std: float = .02,
                 fan: str = 'fan_in',
                 **kwargs):
    """Initialize a linear or conv layer's weight in-place and zero its bias.

    Args:
        x: An ``nn.Linear`` or ``nn.Conv2d`` module to initialize.
        mode: Prefix for the ``torch.nn.init`` function (``''``: ``normal_``, ``'trunc_'``: ``trunc_normal_``, ``'kaiming_'``: ``kaiming_normal_``.)
        std: Standard deviation used when ``mode`` is ``''`` or ``'trunc_'``.
        fan: Fan mode passed to ``kaiming_normal_`` when ``mode='kaiming_'``.
        **kwargs: Additional keyword arguments forwarded to the init function.
    """
    if mode in ('', 'trunc_'): kwargs['std'] = std
    if mode == 'kaiming_': kwargs['mode'] = fan
    getattr(init, f"{mode}normal_")(x.weight, **kwargs)
    if x.bias is not None: nn.init.zeros_(x.bias)


def apply_init(x: Union[nn.Linear, nn.Conv2d],
               general_init: bool = True, zeroB: bool = True, **kwargs):
    """Apply weight initialization to a linear, conv, or LoRA layer in-place.

    For ``LoRALinear``, initializes both the base linear weights and the LoRA
    adapter weights. For plain ``nn.Linear`` / ``nn.Conv2d``, only the base
    weights are initialized. Other module types are silently skipped.

    Args:
        x: Module to initialize.
        general_init: Passed to ``init_lora_`` to choose between standard LoRA
            initialization (Kaiming + zeros) and the custom distribution.
        **kwargs: Forwarded to ``init_linear_`` and ``init_lora_``.
    """
    if isinstance(x, LoRALinear):
        init_lora_(x, general_init=general_init, zeroB=zeroB, **kwargs)
        return
    if not isinstance(x, (nn.Linear, nn.Conv2d)): return
    init_linear_(x, **kwargs)


def init_linear(mode: str = 'trunc_', std: float = .02, fan: str = 'fan_in', **kwargs):
    """Return a callable that initializes a linear/conv/LoRA layer with fixed settings.

    Convenient for use with ``module.apply()``.

    Args:
        mode: Init mode prefix (see ``init_linear_``).
        std: Standard deviation for normal-family initializers.
        fan: Fan mode for Kaiming initialization.

    Returns:
        A partial of ``apply_init`` with ``mode``, ``std``, and ``fan`` bound.
    """
    return partial(apply_init, mode=mode, std=std, fan=fan, **kwargs)


def init_norm_(x: nn.Module, val: float = 1.0):
    """Initialize normalization layer weights to ``val`` and biases to zero in-place.

    Supports ``BatchNorm1d``, ``BatchNorm2d``, ``GroupNorm``, and ``LayerNorm``.
    Other module types are silently skipped.

    Args:
        x: Module to initialize.
        val: Constant value for the weight parameter.
    """
    if not isinstance(x, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)): return
    nn.init.constant_(x.weight, val)
    if x.bias is not None: nn.init.zeros_(x.bias)


def init_norm(val: float = 1.0):
    """Return a callable that initializes normalization layers with a fixed weight value.

    Convenient for use with ``module.apply()``.

    Args:
        val: Constant value to set for the weight parameter.

    Returns:
        A partial of ``init_norm_`` with ``val`` bound.
    """
    return partial(init_norm_, val=val)