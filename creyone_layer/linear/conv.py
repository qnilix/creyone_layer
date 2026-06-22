import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear2d(nn.Linear):

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

class Linear3d(nn.Linear):

    def forward(self, x: torch.Tensor):
        return F.conv3d(x, self.weight[:, :, None, None, None], self.bias)