from typing import Callable, Optional, Union
from torch import nn

from .registry import layer_entrypoint

def create_layer(
        layer_name: Optional[Union[str, nn.Module]],
        layer_family: str = 'any', 
        otherwise: Callable = lambda *a, **b: nn.Identity()) -> Union[nn.Module, Callable]:
    
    if layer_name is None: return otherwise
    if isinstance(layer_name, str): return layer_entrypoint(layer_name, layer_family=layer_family)
    return layer_name