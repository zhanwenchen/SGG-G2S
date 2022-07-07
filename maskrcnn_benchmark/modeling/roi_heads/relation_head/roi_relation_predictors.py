# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from pickle import load as pickle_load
from torch import (
    device as torch_device,
    cat as torch_cat,
    stack as torch_stack,
    no_grad as torch_no_grad,
    from_numpy as torch_from_numpy,
    sigmoid as torch_sigmoid,
    zeros as torch_zeros,
    arange as torch_arange,
    empty_like as torch_empty_like,
    empty as torch_empty,
    float32 as torch_float32,
    mm as torch_mm,
    as_tensor as torch_as_tensor,
    eye as torch_eye,
    tanh as torch_tanh,
    allclose as torch_allclose,
    get_num_threads as torch_get_num_threads,
    add as torch_add,
    mul as torch_mul,
    addcmul as torch_addcmul,
    einsum as torch_einsum,
    matmul as torch_matmul,
    transpose as torch_transpose,
    stack as torch_stack,
)
from torch.utils.benchmark import Timer, Compare
from torch.jit import script as torch_jit_script
from torch.nn import Module, Sequential, Linear, ReLU, GroupNorm
from torch.nn.functional import dropout as F_dropout, binary_cross_entropy_with_logits as F_binary_cross_entropy_with_logits, relu as F_relu, softmax as F_softmax, cross_entropy as F_cross_entropy
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.layers.gcn._utils import adj_normalize
from .model_motifs import FrequencyBias, to_onehot
from .model_transformer import TransformerContext, TransformerEncoder
from .utils_relation import MLP, layer_init_kaiming_normal
from .lrga import LowRankAttention
from .model_ggnn import GGNNContext


@registry.ROI_RELATION_PREDICTOR.register("GBNetPredictor")
class GBNetPredictor(Module):
    def __init__(self, config, in_channels):
        super().__init__()
        self.ggnn = GGNNContext(config)
        self.use_bias = False
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)

        hidden_dim = config.MODEL.ROI_RELATION_HEAD.GBNET.HIDDEN_DIM
        self.obj_dim = 4096
        self.rel_dim = 4096
        self.obj_proj = Linear(self.obj_dim, hidden_dim)
        # layer_init_kaiming_normal(self.obj_proj)
        self.rel_proj = Linear(self.rel_dim, hidden_dim)
        # layer_init_kaiming_normal(self.rel_proj)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, global_image_features=None):
        """
        Args:
            proposals: list[Tensor]: of length batch_size
                       len(proposals) = 16
                       proposal_0 = proposals[0]
                       proposal_0 is BoxList(num_boxes=80, image_width=600, image_height=900, mode=xyxy)
                       len(proposal_0) = 80
                       proposal_0[[0]] is the way to access the boxlist
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
                                          (Pdb) rel_pair_idxs[0].size()
                                          torch.Size([156, 2])

            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
                                     (Pdb) union_features.size()
                                     torch.Size([3009, 4096])
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        '''
        len(proposals) = 8
        len(rel_pair_idxs) = 8
        len(rel_labels) = 8
        len(rel_binarys) = 8
        roi_features.shape = torch.Size([94, 4096])
        union_features.shape = torch.Size([1118, 4096])
        global_image_features.shape = torch.Size([8, 4096])
        '''
        roi_features = self.obj_proj(roi_features) # torch.Size([85, 4096]) => torch.Size([85, 1024])
        union_features = self.rel_proj(union_features) # torch.Size([1138, 4096]) => torch.Size([1138, 1024])
        #
        # if proposals is None:
        #     print(f'proposals is None')
        #     breakpoint()
        #
        # if rel_pair_idxs is None:
        #     print(f'rel_pair_idxs is None')
        #     breakpoint()
        # num_objs = [len(proposal) for proposal in proposals]
        # num_rels = [len(rel_pair_idx) for rel_pair_idx in rel_pair_idxs]
        # roi_features_images = roi_features.split(num_objs)
        # union_features_images = union_features.split(num_rels)
        #
        # obj_dists, rel_dists = self.gbnet_context(roi_features_images, union_features_images)
        #
        # obj_dists = obj_dists.split(num_objs, dim=0)
        # rel_dists = rel_dists.split(num_rels, dim=0)
        #
        # return obj_dists, rel_dists, {}

        # if self.union_single_not_match:
        #     union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.ggnn(proposals, rel_pair_idxs, roi_features, union_features, device=None, debug=True)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append(torch_stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
