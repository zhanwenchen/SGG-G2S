# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import float16 as torch_float16, zeros as torch_zeros, ones as torch_ones
from torch.nn import Module

class FrozenBatchNorm2d(Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch_ones(n))
        self.register_buffer("bias", torch_zeros(n))
        self.register_buffer("running_mean", torch_zeros(n))
        self.register_buffer("running_var", torch_ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch_float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias
