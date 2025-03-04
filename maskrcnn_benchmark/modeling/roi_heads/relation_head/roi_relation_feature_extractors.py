# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import arange as torch_arange, cat as torch_cat, stack as torch_stack
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union, boxlist_intersection
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from maskrcnn_benchmark.modeling.roi_heads.attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor



@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("RelationFeatureExtractor")
class RelationFeatureExtractor(nn.Module):
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
            self.spatial_fc = nn.Sequential(*[make_fc(input_size, out_dim//2), nn.ReLU(inplace=True),
                                              make_fc(out_dim//2, out_dim), nn.ReLU(inplace=True),
                                            ])

        # union rectangle size
        self.rect_size = resolution * 4 -1
        self.rect_conv = nn.Sequential(*[
            nn.Conv2d(2, in_channels //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
            ])


    def forward(self, x, proposals, rel_pair_idxs=None, return_indices=None):
        # import pdb; pdb.set_trace()
        device = x[0].device
        union_proposals = []
        rect_inputs = []
        # num_proposals = [len(proposal) for proposal in proposals]
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]
            union_proposals.append(boxlist_union(head_proposal, tail_proposal))

            # use range to construct rectangle, sized (rect_size, rect_size)
            num_rel = len(rel_pair_idx)
            dummy_x_range = torch_arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size, self.rect_size)
            dummy_y_range = torch_arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size, self.rect_size)
            # resize bbox to the scale rect_size
            head_proposal = head_proposal.resize((self.rect_size, self.rect_size))
            tail_proposal = tail_proposal.resize((self.rect_size, self.rect_size))
            head_rect = ((dummy_x_range >= head_proposal.bbox[:,0].floor().view(-1,1,1).long()) & \
                        (dummy_x_range <= head_proposal.bbox[:,2].ceil().view(-1,1,1).long()) & \
                        (dummy_y_range >= head_proposal.bbox[:,1].floor().view(-1,1,1).long()) & \
                        (dummy_y_range <= head_proposal.bbox[:,3].ceil().view(-1,1,1).long())).float()
            tail_rect = ((dummy_x_range >= tail_proposal.bbox[:,0].floor().view(-1,1,1).long()) & \
                        (dummy_x_range <= tail_proposal.bbox[:,2].ceil().view(-1,1,1).long()) & \
                        (dummy_y_range >= tail_proposal.bbox[:,1].floor().view(-1,1,1).long()) & \
                        (dummy_y_range <= tail_proposal.bbox[:,3].ceil().view(-1,1,1).long())).float()

            # (num_rel, 4, rect_size, rect_size) # torch.Size([651, 2, 27, 27]), torch.Size([110, 2, 27, 27])
            rect_inputs.append(torch_stack((head_rect, tail_rect), dim=1))

        # rectangle feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        rect_inputs = torch_cat(rect_inputs, dim=0) # original: rect_inputs = 16 * torch.Size([651, 2, 27, 27]), [650, ...], [110, ...], ... => torch.Size([5049, 2, 27, 27])
        rect_features = self.rect_conv(rect_inputs) # torch.Size([5049, 256, 7, 7])

        # union visual feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        union_vis_features = self.feature_extractor.pooler(x, union_proposals) # union_proposals: 16 * [651..., 650..., 110...] # union_vis_features: torch.Size([5049, 256, 7, 7]) # TODO: need to borrow pooler's 5 layers to 1 reduction. so have a global union feature pooler
        # like GLCNet: Yong Liu, Ruiping Wang, S. Shan, and Xilin Chen. Structure inference net: . Zellers (MotifNET) models global context (between boxes) by LSTM (but as a state, not from visual features). I can do it by a Linear from visual features.
        # merge two parts
        if self.separate_spatial: # False
            region_features = self.feature_extractor.forward_without_pool(union_vis_features)
            spatial_features = self.spatial_fc(rect_features.view(rect_features.size(0), -1))
            union_features = (region_features, spatial_features)
        else:
            union_features = union_vis_features + rect_features
            union_features = self.feature_extractor.forward_without_pool(union_features) # (total_num_rel, out_channels) torch.Size([5049, 4096])

        if self.cfg.MODEL.ATTRIBUTE_ON:
            union_att_features = self.att_feature_extractor.pooler(x, union_proposals)
            union_features_att = union_att_features + rect_features
            union_features_att = self.att_feature_extractor.forward_without_pool(union_features_att)
            union_features = torch_cat((union_features, union_features_att), dim=-1)

        # if return_indices is True:
            # return union_features, indices
        return union_features


def make_roi_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
