# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import as_tensor as torch_as_tensor, int64 as torch_int64, nonzero as torch_nonzero
from torch.nn import SmoothL1Loss
from torch.nn.functional import cross_entropy as F_cross_entropy
# from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
# from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
#     BalancedPositiveNegativeSampler
# )


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, cls_agnostic_bbox_reg=False):
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        # self.one = torch_as_tensor(1, device='cuda', dtype=torch_int64)
        self.one_thru_three = torch_as_tensor([0, 1, 2, 3], device='cuda', dtype=torch_int64)
        self.four_thru_seven = torch_as_tensor([4, 5, 6, 7], device='cuda', dtype=torch_int64)
        self.smooth_l1_loss = SmoothL1Loss(reduction='sum', beta=1.0)

    def assign_label_to_proposals(self, proposals, targets):
        for img_idx, (target, proposal) in enumerate(zip(targets, proposals)):
            match_quality_matrix = boxlist_iou(target, proposal)
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            # Fast RCNN only need "labels" field for selecting the targets
            target = target.copy_with_fields(["labels", "attributes"])
            matched_targets = target[matched_idxs.clamp(min=0)]

            labels_per_image = matched_targets.get_field("labels").to(dtype=torch_int64)
            attris_per_image = matched_targets.get_field("attributes").to(dtype=torch_int64)

            matched_idxs_lt_0 = matched_idxs < 0
            labels_per_image[matched_idxs_lt_0] = 0
            attris_per_image[matched_idxs_lt_0, :] = 0
            proposals[img_idx].add_field("labels", labels_per_image)
            proposals[img_idx].add_field("attributes", attris_per_image)
        return proposals


    def __call__(self, class_logits, box_regression, proposals):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])
            proposals (list[BoxList])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        # device = class_logits.device

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat([proposal.get_field("regression_targets") for proposal in proposals], dim=0)

        classification_loss = F_cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch_nonzero(labels > 0).squeeze_(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = self.four_thru_seven.detach().clone()
        else:
            map_inds = self.one_thru_three.detach().clone().add(labels_pos.unsqueeze_(-1).mul_(4))

        box_loss = self.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset.unsqueeze(-1), map_inds],
            regression_targets[sampled_pos_inds_subset],
        ).div_(labels.numel())
        # box_loss = smooth_l1_loss(
        #     box_regression[sampled_pos_inds_subset.unsqueeze(-1), map_inds],
        #     regression_targets[sampled_pos_inds_subset],
        #     self.one,
        # )

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(cls_agnostic_bbox_reg)

    return loss_evaluator
