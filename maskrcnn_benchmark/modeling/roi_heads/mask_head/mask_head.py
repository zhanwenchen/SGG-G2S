# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_mask_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        """
        training with 80*16 images:
        1.  (Pdb) features.size()
            torch.Size([139, 4096])
        2. len(proposals) = 16.
            (Pdb) proposals[0]
            BoxList(num_boxes=10, image_width=800, image_height=600, mode=xyxy)
        3. (Pdb) targets
            [BoxList(num_boxes=10, image_width=800, image_height=600, mode=xyxy), BoxList(num_boxes=5, image_width=896, image_height=600, mode=xyxy), BoxList(num_boxes=15, image_width=895, image_height=600, mode=xyxy), BoxList(num_boxes=13, image_width=900, image_height=600, mode=xyxy), BoxList(num_boxes=7, image_width=800, image_height=600, mode=xyxy), BoxList(num_boxes=10, image_width=800, image_height=600, mode=xyxy), BoxList(num_boxes=10, image_width=900, image_height=600, mode=xyxy), BoxList(num_boxes=10, image_width=800, image_height=600, mode=xyxy), BoxList(num_boxes=8, image_width=1000, image_height=472, mode=xyxy), BoxList(num_boxes=3, image_width=800, image_height=600, mode=xyxy), BoxList(num_boxes=6, image_width=646, image_height=600, mode=xyxy), BoxList(num_boxes=5, image_width=1000, image_height=538, mode=xyxy), BoxList(num_boxes=11, image_width=906, image_height=600, mode=xyxy), BoxList(num_boxes=10, image_width=892, image_height=600, mode=xyxy), BoxList(num_boxes=10, image_width=900, image_height=600, mode=xyxy), BoxList(num_boxes=6, image_width=999, image_height=562, mode=xyxy)]
        """

        # breakpoint()
        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            # breakpoint()
            proposals, positive_inds = keep_only_positive_boxes(proposals)
                # (Pdb) proposals
                # [BoxList(num_boxes=10, image_width=800, image_height=600, mode=xyxy), BoxList(num_boxes=5, image_width=896, image_height=600, mode=xyxy), BoxList(num_boxes=15, image_width=895, image_height=600, mode=xyxy), BoxList(num_boxes=13, image_width=900, image_height=600, mode=xyxy), BoxList(num_boxes=7, image_width=800, image_height=600, mode=xyxy), BoxList(num_boxes=10, image_width=800, image_height=600, mode=xyxy), BoxList(num_boxes=10, image_width=900, image_height=600, mode=xyxy), BoxList(num_boxes=10, image_width=800, image_height=600, mode=xyxy)]
                # (Pdb) len(proposals)
                # 8

                # (Pdb) len(positive_inds)
                # 8
                # (Pdb) positive_inds[0].size()
                # torch.Size([10])
                # (Pdb) positive_inds[1].size()
                # torch.Size([5])

            # breakpoint()
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            # breakpoint()
            x = features
            # (Pdb) features.size()
            # torch.Size([80, 4096])
            # breakpoint()
            x = x[torch.cat(positive_inds, dim=0)]
            # (Pdb) torch.cat(positive_inds, dim=0).size()
            # torch.Size([80])
        else:
            x = self.feature_extractor(features, proposals)
            # breakpoint()
        # (Pdb) x.size()
        # torch.Size([95, 256, 14, 14])
        mask_logits = self.predictor(x)
        # breakpoint()
        # (Pdb) mask_logits.size()
        # torch.Size([95, 151, 28, 28])

        if not self.training:
            # breakpoint()
            result = self.post_processor(mask_logits, proposals)
            # breakpoint()
            return x, result, {}

        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)
        # breakpoint()

        return x, all_proposals, dict(loss_mask=loss_mask)


def build_roi_mask_head(cfg, in_channels):
    return ROIMaskHead(cfg, in_channels)
