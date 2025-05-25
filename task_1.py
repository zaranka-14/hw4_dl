import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.eps = 1e-6
        self.normalized_shape = torch.Size(normalized_shape)
        self.param = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, a):
        dims = tuple(-(i + 1) for i in range(len(self.normalized_shape)))
        rms = torch.sqrt(torch.pow(a, 2).mean(dim=dims, keepdim=True)) + self.eps
        return self.param * a * 1.0 / rms
