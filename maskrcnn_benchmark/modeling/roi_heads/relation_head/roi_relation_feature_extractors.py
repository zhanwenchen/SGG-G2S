# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import (
    arange as torch_arange,
    cat as torch_cat,
    stack as torch_stack,
    int16 as torch_int16,
)
from torch.nn import Module, Sequential, Conv2d, ReLU, BatchNorm2d, MaxPool2d
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from maskrcnn_benchmark.modeling.roi_heads.attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor



@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("RelationFeatureExtractor")
class RelationFeatureExtractor(Module):
    """
    Heads for Motifs for relation triplet classification
    """
    def __init__(self, cfg, in_channels):
        super(RelationFeatureExtractor, self).__init__()
        self.cfg = cfg.clone()
        # should corresponding to obj_feature_map function in neural-motifs
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pool_all_levels = cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS

        if cfg.MODEL.ATTRIBUTE_ON:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True, cat_all_levels=pool_all_levels)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels * 2
        else:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels

        # separete spatial
        self.separate_spatial = self.cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        if self.separate_spatial:
            input_size = self.feature_extractor.resize_channels
            out_dim = self.feature_extractor.out_channels
            self.spatial_fc = Sequential(*[make_fc(input_size, out_dim//2), ReLU(inplace=True),
                                           make_fc(out_dim//2, out_dim), ReLU(inplace=True),
                                          ])

        # union rectangle size
        self.rect_size = resolution * 4 -1
        self.rect_conv = Sequential(*[
            Conv2d(2, in_channels //2, kernel_size=7, stride=2, padding=3, bias=True),
            ReLU(inplace=True),
            BatchNorm2d(in_channels//2, momentum=0.01),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
            Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            ReLU(inplace=True),
            BatchNorm2d(in_channels, momentum=0.01),
        ])

        self.dummy_range = torch_arange(self.rect_size, device='cuda', dtype=torch_int16).view(1, self.rect_size)


    def forward(self, x, proposals, rel_pair_idxs=None):
        union_proposals = []
        rect_inputs = []
        self_rect_size = self.rect_size

        dummy_range = self.dummy_range
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]
            del proposal, rel_pair_idx
            union_proposal = boxlist_union(head_proposal, tail_proposal)
            union_proposals.append(union_proposal)
            del union_proposal

            # resize bbox to the scale rect_size
            head_proposal = head_proposal.resize((self_rect_size, self_rect_size)).bbox
            tail_proposal = tail_proposal.resize((self_rect_size, self_rect_size)).bbox

            # use range to construct rectanglesized (rect_size, rect_size)
            head_rect = ((dummy_range >= head_proposal[:, 0, None].floor().short()).unsqueeze(1) & \
                         (dummy_range <= head_proposal[:, 2, None].ceil().short()).unsqueeze(1) & \
                         (dummy_range >= head_proposal[:, 1, None].floor().short()).unsqueeze(2) & \
                         (dummy_range <= head_proposal[:, 3, None].ceil().short()).unsqueeze(2))
            del head_proposal
            tail_rect = ((dummy_range >= tail_proposal[:, 0, None].floor().short()).unsqueeze(1) & \
                         (dummy_range <= tail_proposal[:, 2, None].ceil().short()).unsqueeze(1) & \
                         (dummy_range >= tail_proposal[:, 1, None].floor().short()).unsqueeze(2) & \
                         (dummy_range <= tail_proposal[:, 3, None].ceil().short()).unsqueeze(2))

            del tail_proposal
            # (num_rel, 4, rect_size, rect_size) # torch.Size([651, 2, 27, 27]), torch.Size([110, 2, 27, 27])
            rect_inputs.append(torch_stack((head_rect, tail_rect), dim=1))
            del head_rect, tail_rect
        del rel_pair_idxs
        # rectangle feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        rect_inputs = torch_cat(rect_inputs, dim=0).float() # original: rect_inputs = 16 * torch.Size([651, 2, 27, 27]), [650, ...], [110, ...], ... => torch.Size([5049, 2, 27, 27])
        rect_features = self.rect_conv(rect_inputs) # torch.Size([5049, 256, 7, 7])
        del rect_inputs
        # union visual feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        union_vis_features = self.feature_extractor.pooler(x, union_proposals) # union_proposals: 16 * [651..., 650..., 110...] # union_vis_features: torch.Size([5049, 256, 7, 7]) # TODO: need to borrow pooler's 5 layers to 1 reduction. so have a global union feature pooler
        if not self.cfg.MODEL.ATTRIBUTE_ON:
            del x, union_proposals
        # like GLCNet: Yong Liu, Ruiping Wang, S. Shan, and Xilin Chen. Structure inference net: . Zellers (MotifNET) models global context (between boxes) by LSTM (but as a state, not from visual features). I can do it by a Linear from visual features.
        # merge two parts
        if self.separate_spatial: # False
            region_features = self.feature_extractor.forward_without_pool(union_vis_features)
            spatial_features = self.spatial_fc(rect_features.view(rect_features.size(0), -1))
            union_features = (region_features, spatial_features)
        else:
            union_features = union_vis_features + rect_features
            if not self.cfg.MODEL.ATTRIBUTE_ON: del union_vis_features, rect_features
            union_features = self.feature_extractor.forward_without_pool(union_features) # (total_num_rel, out_channels) torch.Size([5049, 4096])

        if self.cfg.MODEL.ATTRIBUTE_ON:
            union_att_features = self.att_feature_extractor.pooler(x, union_proposals)
            union_features_att = union_att_features + rect_features
            union_features_att = self.att_feature_extractor.forward_without_pool(union_features_att)
            union_features = torch_cat((union_features, union_features_att), dim=-1)

        return union_features


def make_roi_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
