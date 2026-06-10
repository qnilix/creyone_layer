# creyone_layer (Beta Package for CreYoNe)

Create Your Network (a.k.a. CreYoNe) is a utility tools for making DNN module via torch. This repository provides Layer-sized building blocks for deep learning.

## Installation

```bash
pip install creyone-layer
```

From source:

```bash
git clone https://github.com/qaiLN/creyone.git
cd creyone_layer
pip install -e .
```

## Quick Start

### `CNNBlockCfg` — configurable CNN block factory

```python
from creyone_layer import CNNBlockCfg

# Default: 2D Conv + BatchNorm2d + ReLU with auto-padding
cfg = CNNBlockCfg()

# ConvNormAct block (Conv → Norm → Act)
block = cfg.block_module(in_dim=32, out_dim=64, k=3)

# Build each layer individually
conv = cfg.conv_layer(32, 64, 3)   # Conv2d 3×3, auto-padded
norm = cfg.norm_layer(64)           # BatchNorm2d
act  = cfg.act_layer()              # ReLU
pool = cfg.pool_layer(2)            # MaxPool2d k=2

# Customize config per dimension / activation
cfg_3d = CNNBlockCfg(tensor_dims=3, act_name='gelu', norm_name='layer')
block_3d = cfg_3d.block_module(16, 32, k=3)

# Per-call overrides (don't change cfg itself)
dw_block = cfg.block_module(64, 64, k=3, name='depthwise')
```

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
