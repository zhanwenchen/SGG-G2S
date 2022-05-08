# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from maskrcnn_benchmark._C import nms as nms_cuda
from apex.amp import float_function as amp_float_function

# Only valid with fp32 inputs - give AMP the hint
nms = amp_float_function(nms_cuda)

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
