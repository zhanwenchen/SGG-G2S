# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import abs as torch_abs, where as torch_where


# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch_abs(input - target)
    cond = n < beta
    loss = torch_where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
