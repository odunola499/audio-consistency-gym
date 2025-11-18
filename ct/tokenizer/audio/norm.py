from typing import List, Union

import torch
from torch import nn
from torch.nn import functional as F


class ConvLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape):
        super().__init__(normalized_shape)

    def forward(self, x):
        # Shape of [B, C, T] -> [B, T, C] so norm happens along channels
        x = x.transpose(1, 2)
        x = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float(),
            self.bias.float(),
            self.eps,
        ).type_as(x)
        return x.transpose(1, 2)


class ConvRMSNorm(nn.RMSNorm):
    def __init__(
        self, dim, eps: float = 1e-5, elementwise_affine=True, weight_shape=None
    ):
        super().__init__(
            normalized_shape=dim, eps=eps, elementwise_affine=elementwise_affine
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.rms_norm(
            x.float(), self.normalized_shape, self.weight.float(), self.eps
        ).type_as(x)
        return x.transpose(1, 2)
