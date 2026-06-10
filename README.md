# creyone_layer (Beta Package for CreYoNe)

Create Your Network (a.k.a. CreYoNe) is a utility tools for making DNN module via torch. This repository provides Layer-sized building blocks for deep learning.

## Installation

```bash
pip install creyone-layer
```

From source:

```bash
git clone https://github.com/qnilix/creyone_layer.git
cd creyone_layer
pip install -e .
```

## Quick Start

### Layer registry — `create_layer`

```python
from creyone_layer import create_layer

relu = create_layer('relu', 'act')(inplace=True)()
bn   = create_layer('batch', 'norm')(dim=2, eps=1e-5, mom=0.1)(64)
conv = create_layer('base',  'conv')(dim=2, optional='ap')(32, 64, 3)
pool = create_layer('max',   'pool')(dim=2, optional='ap')(2)
```

## Registered Layers

| Family | Names                                                                               |
| ------ | ----------------------------------------------------------------------------------- |
| `conv` | `base`, `depthwise`                                                                 |
| `norm` | `batch`, `layer`                                                                    |
| `act`  | `relu`, `relu6`, `gelu`, `quickgelu`, `sigmoid`, `hardsig`, `hardswish`, `swisheff` |
| `pool` | `max`, `avg`                                                                        |

### `conv` options (passed via `optional='...'`, `+`-separated)

| Flag   | Effect                                              |
| ------ | --------------------------------------------------- |
| `ap`   | auto-pad — output spatial size matches input        |
| `dw`   | depthwise — `groups = in_channels`                  |
| `grid` | stride = kernel size (grid-like sampling)           |
| `ar`   | AutoReshape — accepts `(B, H*W, C)` token sequences |
