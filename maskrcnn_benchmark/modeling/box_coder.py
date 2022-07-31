# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from math import log as math_log
from torch import (
    clamp as torch_clamp,
    stack as torch_stack,
    log as torch_log,
    exp as torch_exp,
    empty_like as torch_empty_like,
    float32 as torch_float32,
    as_tensor as torch_as_tensor,
)
from torch.jit import script as torch_jit_script


@torch_jit_script
def encode_jit(reference_boxes, proposals, weights):
    """
    Encode a set of proposals with respect to some
    reference boxes

    Arguments:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
    """
    wx, wy, ww, wh = weights[0], weights[1], weights[2], weights[3]
    del weights

    TO_REMOVE = 1  # TODO remove
    ex_widths = (proposals[:, 2] - proposals[:, 0]).add_(TO_REMOVE)
    ex_heights = (proposals[:, 3] - proposals[:, 1]).add_(TO_REMOVE)
    ex_ctr_x = ex_widths.mul(0.5).add_(proposals[:, 0])
    ex_ctr_y = ex_heights.mul(0.5).add_(proposals[:, 1])
    del proposals

    gt_widths = (reference_boxes[:, 2] - reference_boxes[:, 0]).add_(TO_REMOVE)
    gt_heights = (reference_boxes[:, 3] - reference_boxes[:, 1]).add_(TO_REMOVE)
    gt_ctr_x = gt_widths.mul(0.5).add_(reference_boxes[:, 0])
    gt_ctr_y = gt_heights.mul(0.5).add_(reference_boxes[:, 1])

    # addcdiv
    targets_dx = gt_ctr_x.sub_(ex_ctr_x).div_(ex_widths).mul_(wx)
    del ex_ctr_x, wx
    targets_dy = gt_ctr_y.sub_(ex_ctr_y).div_(ex_heights).mul_(wy)
    del ex_ctr_y, wy
    targets_dw = gt_widths.div_(ex_widths).log_().mul_(ww)
    del ex_widths, ww
    targets_dh = gt_heights.div_(ex_heights).log_().mul_(wh)
    del ex_heights, wh
    return torch_stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)


@torch_jit_script
def decode_jit(rel_codes, boxes, weights, bbox_xform_clip):
    """
    From a set of original boxes and encoded relative box offsets,
    get the decoded boxes.

    Arguments:
        rel_codes (Tensor): encoded boxes
        boxes (Tensor): reference boxes.
    """
    wx, wy, ww, wh = weights[0], weights[1], weights[2], weights[3]
    dtype = rel_codes.dtype
    boxes = boxes.type(dtype)

    TO_REMOVE = 1  # TODO remove
    widths = (boxes[:, 2] - boxes[:, 0]).add_(TO_REMOVE)
    heights = (boxes[:, 3] - boxes[:, 1]).add_(TO_REMOVE)
    ctr_x = widths.mul(0.5).add_(boxes[:, 0])
    ctr_y = heights.mul(0.5).add_(boxes[:, 1])
    # There's some opportunity to optimize these below and later:
    # pred_ctr_x = dx.mul_(widths.unsqueeze_()).add_(ctr_x.unsqueeze_())
    # Be careful of pointers.

    dx = rel_codes[:, 0::4] / wx
    dy = rel_codes[:, 1::4] / wy
    dw = rel_codes[:, 2::4] / ww
    dh = rel_codes[:, 3::4] / wh

    # Prevent sending too large values into torch.exp()
    # dw = torch_clamp(dw, max=self.bbox_xform_clip)
    # dh = torch_clamp(dh, max=self.bbox_xform_clip)
    dw.clamp_(max=bbox_xform_clip)
    dh.clamp_(max=bbox_xform_clip)


    pred_ctr_x = dx.mul_(widths.unsqueeze_(-1)).add_(ctr_x.unsqueeze_(-1))
    del ctr_x
    pred_ctr_y = dy.mul_(heights.unsqueeze_(-1)).add_(ctr_y.unsqueeze_(-1))
    del ctr_y
    pred_w = dw.float().exp_().type(dtype).mul_(widths).mul_(0.5)
    pred_h = dh.float().exp_().type(dtype).mul_(heights).mul_(0.5)

    pred_boxes = torch_empty_like(rel_codes, dtype=dtype, device=rel_codes.device)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - pred_h
    # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 2::4] = (pred_ctr_x + pred_w).sub_(1)
    # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] = (pred_ctr_y + pred_h).sub_(1)

    return pred_boxes


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """
    def __init__(self, weights, bbox_xform_clip=math_log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = torch_as_tensor(weights, dtype=torch_float32, device='cuda')
        self.bbox_xform_clip = torch_as_tensor(bbox_xform_clip, dtype=torch_float32, device='cuda')

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """

        # wx, wy, ww, wh = self.weights
        # dtype, device = reference_boxes.dtype, reference_boxes.device
        # weights = torch_as_tensor(self.weights, dtype=dtype, device=device)
        return encode_jit(reference_boxes, proposals, self.weights.to(reference_boxes, non_blocking=True))

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        # wx, wy, ww, wh = self.weights
        dtype, device = rel_codes.dtype, rel_codes.device
        # weights = torch_as_tensor(self.weights, dtype=dtype, device=device)
        # bbox_xform_clip = self.bbox_xform_clip
        # bbox_xform_clip = torch_as_tensor(self.bbox_xform_clip, dtype=dtype, device=device)
        return decode_jit(rel_codes, boxes, self.weights.to(dtype=dtype, device=device, non_blocking=True), self.bbox_xform_clip.to(dtype=dtype, device=device, non_blocking=True))
