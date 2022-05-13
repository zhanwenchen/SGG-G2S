# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from maskrcnn_benchmark._C import nms as _C_nms

from apex.amp import float_function

# Only valid with fp32 inputs - give AMP the hint
nms = float_function(_C_nms)

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
