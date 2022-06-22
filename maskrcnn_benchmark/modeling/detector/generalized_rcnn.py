# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, logger=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            (Pdb) images.tensors.size()
            torch.Size([16, 3, 1024, 608])

            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images) # (Pdb) images.tensors.size() torch.Size([16, 3, 608, 1024])
        boxes_global = [BoxList([[0, 0, *image_size]], image_size, device=images.tensors.device) for image_size in images.image_sizes]
        features = self.backbone(images.tensors)
        # (Pdb) len(features)
        # 5
        # (Pdb) features[0].size()
        # torch.Size([16, 256, 256, 152])
        # (Pdb) features[1].size()
        # torch.Size([16, 256, 128, 76])
        # (Pdb) features[2].size()
        # torch.Size([16, 256, 64, 38])
        # (Pdb) features[3].size()
        # torch.Size([16, 256, 32, 19])
        # (Pdb) features[4].size()
        # torch.Size([16, 256, 16, 10])

        proposals, proposal_losses = self.rpn(images, features, targets)
        del images

        # boxlist
        if self.roi_heads:
            _, result, detector_losses = self.roi_heads(features, proposals, targets, logger, boxes_global=boxes_global)
            del boxes_global
        else:
            # RPN-only models don't have roi_heads
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            if not self.cfg.MODEL.RELATION_ON:
                # During the relationship training stage, the rpn_head should be fixed, and no loss.
                losses.update(proposal_losses)
            return losses

        return result
