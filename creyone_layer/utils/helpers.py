""" Layer/Module Helpers

Copyright 2020 Ross Wightman (original, Apache-2.0)
Modifications Copyright 2026 Rinka Kiriyama
"""
from itertools import repeat
import collections.abc

from typing import Callable


def ntuple(n) -> Callable:
    """Return a function that converts input to an n-tuple.

    Scalar values are repeated n times, while iterables are converted to tuples.
    Strings are treated as scalars to avoid character-level splitting.

    Args:
        n: Target tuple length.

    Returns:
        Function that converts input to n-tuple.
    """
    def parse(x) -> tuple:
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            t = tuple(x)
            if len(t) != n:
                raise ValueError(f"expected {n}-tuple, got length {len(t)}")
            return t
        return tuple(repeat(x, n))
    return parse