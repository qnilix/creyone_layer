# Copyright (c) 2026 QNiLix
# Licensed under the MIT License (MIT). 
#
# Portions based on loralib by Microsoft Corporation
# (https://github.com/microsoft/LoRA), licensed under the MIT License.

""" Linear layer with Low-Rank Adaptation (LoRA)

Copyright 2026 Rinka Kiriyama。
Licensed under the MIT License (MIT). 

Portions based on loralib by Microsoft Corporation
(https://github.com/microsoft/LoRA), licensed under the MIT License.

References:
    Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S.,
    Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large
    Language Models. *International Conference on Learning Representations
    (ICLR 2022)*. https://arxiv.org/abs/2106.09685
"""

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Linear):
    """Linear layer with Low-Rank Adaptation (LoRA).

    Augments a frozen ``nn.Linear`` with trainable low-rank matrices A and B
    so that the effective weight becomes ``W + B @ A``.  Only A and B are
    updated during fine-tuning; the base weight is left frozen.

    The forward computation is::

        h = x @ W.T + (dropout(x) @ A.T @ B.T) * (alpha / r)

    Args:
        in_features:    Size of each input sample.
        out_features:   Size of each output sample.
        lora_r:         Rank of the low-rank decomposition.  When 0 the layer
                        behaves identically to a plain ``nn.Linear``.
        alpha:          LoRA scaling factor.  The effective scale applied to the
                        LoRA contribution is ``alpha / lora_r``.
        drop:           Dropout probability applied to the input before the
                        LoRA branch.
        fan_in_fan_out: Set ``True`` when the base weight is stored as
                        ``(in_features, out_features)`` (e.g. GPT-2 Conv1D)
                        instead of the default PyTorch layout.
        **kwargs:       Additional keyword arguments forwarded to ``nn.Linear``
                        (e.g. ``bias``, ``device``, ``dtype``).
    """

    def __init__(self, in_features: int, out_features: int,
                 lora_r: int = 0, alpha: float = 1.0, drop: float = .0,
                 fan_in_fan_out: bool = False, **kwargs):
        super().__init__(in_features, out_features, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        self.drop = nn.Dropout(p=drop)
        self._r = lora_r
        if lora_r > 0:
            # A: initialised with Kaiming uniform so gradients flow from step 1.
            # B: initialised to zero so the LoRA delta starts at zero.
            self.adwA = nn.Parameter(self.weight.new_empty((lora_r, in_features)))
            self.adwB = nn.Parameter(self.weight.new_zeros((out_features, lora_r)))
            nn.init.kaiming_uniform_(self.adwA, a=5 ** 0.5)
            self.scaling = alpha / lora_r
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the LoRA-augmented linear transformation.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        def T(w): return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)
        if self._r > 0:
            result += (self.drop(x) @ self.adwA.transpose(0, 1) @ self.adwB.transpose(0, 1)) * self.scaling
        return result
