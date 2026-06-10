import sys, warnings
from collections import defaultdict
from typing import Any, Callable, Optional


_module_to_layers: dict[str, set[str]] = defaultdict(set)  # dict of sets to check membership of model in module
_layer_to_module: dict[str, dict[str, str]] = defaultdict(dict)  # mapping of model names to module names
_layer_entrypoints: dict[str, dict[str, Callable[..., Any]]] = defaultdict(dict)  # mapping of model names to architecture entrypoint fns


def register_layer(layer_family: str = 'any'):

    def _register_layer(fn: Callable[..., any]) -> Callable[..., any]:

        # lookup containing module
        mod = sys.modules[fn.__module__]
        module_name = fn.__module__.split('.')[-1]

        # add model to __all__ in module
        layer_name = fn.__name__
        if not hasattr(mod, '__all__'): mod.__all__ = []
        mod.__all__.append(layer_name)

        # add entries to registry dict/sets
        if layer_name in _layer_entrypoints[layer_family]:
            warnings.warn(
                f'Overwriting {layer_name} in registry with {fn.__module__}.{layer_name}. This is because the name being '
                'registered conflicts with an existing name. Please check if this is not expected.',
                stacklevel=2,
            )
        _layer_entrypoints[layer_family][layer_name] = fn
        _layer_to_module[layer_family][layer_name] = module_name
        _module_to_layers[module_name].add(layer_name)

        return fn
    
    return _register_layer


def layer_entrypoint(layer_name: str, 
                     layer_family: Optional[str] = None, 
                     module_filter: Optional[str] = None) -> Callable[..., any]:
    """Fetch a model entrypoint for specified model name
    """
    if module_filter and layer_name not in _module_to_layers.get(module_filter, set()):
        raise RuntimeError(f'Model ({layer_name} not found in module {module_filter}.)')
    if layer_family is not None and layer_name in _layer_entrypoints[layer_family]: 
        return _layer_entrypoints[layer_family][layer_name]
    if layer_name not in _layer_entrypoints['any']:
        raise RuntimeError(f'Layer [{layer_name}] not found in {layer_family} and `any`.')
    return _layer_entrypoints['any'][layer_name]