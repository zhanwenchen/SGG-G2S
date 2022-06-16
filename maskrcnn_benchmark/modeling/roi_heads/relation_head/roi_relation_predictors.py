# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from torch import (
    cat as torch_cat, no_grad as torch_no_grad,
    from_numpy as torch_from_numpy,
    as_tensor as torch_as_tensor,
    empty_like as torch_empty_like, empty as torch_empty,
    float32 as torch_float32,
)
from torch.nn import Module, Linear, ModuleList, Sequential, GroupNorm, ReLU
from torch.nn.functional import dropout as F_dropout, relu as F_relu
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.layers.gcn._utils import adj_normalize
from .model_motifs import FrequencyBias
# from .model_transformer import TransformerContext, TransformerEncoder
from .model_transformer import TransformerContext
from .utils_relation import layer_init_kaiming_normal
from .lrga import LowRankAttention


@registry.ROI_RELATION_PREDICTOR.register("TransformerTransferGSCPredictor")
class TransformerTransferGSCPredictor(Module):
    def __init__(self, config, in_channels):
        super(TransformerTransferGSCPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE

        assert in_channels is not None
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.with_knowdist = False
        self.devices = config.MODEL.DEVICE
        self.with_transfer = config.MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER
        self.with_cleanclf = config.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER
        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.val_alpha = config.MODEL.ROI_RELATION_HEAD.VAL_ALPHA

        # module construct
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.v_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM
        self.k_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.inner_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.num_head = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.edge_layer = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER

        # module construct
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.post_emb = Linear(self.hidden_dim, self.hidden_dim * 2)
        # layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init_kaiming_normal(self.post_emb)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init_kaiming_normal(self.up_dim)
        else:
            self.union_single_not_match = False

        self.post_cat = Linear(self.hidden_dim * 2, self.pooling_dim)
        layer_init_kaiming_normal(self.post_cat)

        # LRGA stuff
        self.time_step_num = config.MODEL.ROI_RELATION_HEAD.LRGA.TIME_STEP_NUM
        self.k = config.MODEL.ROI_RELATION_HEAD.LRGA.K
        self.num_groups = config.MODEL.ROI_RELATION_HEAD.LRGA.NUM_GROUPS
        self.dropout = config.MODEL.ROI_RELATION_HEAD.LRGA.DROPOUT
        self.in_channels = self.pooling_dim
        self.hidden_channels = self.pooling_dim
        self.out_channels = self.pooling_dim

        # self.attention = ModuleList()
        # self.dimension_reduce = ModuleList()
        self.attention_1 = LowRankAttention(self.k, self.in_channels, self.dropout)
        self.dimension_reduce_1 = Sequential(Linear(2*self.k + self.hidden_channels, self.hidden_channels), ReLU())
        self.attention_2 = LowRankAttention(self.k, self.hidden_channels, self.dropout)
        self.dimension_reduce_2 = Sequential(Linear(2*self.k + self.hidden_channels, self.out_channels))
        self.gn = GroupNorm(self.num_groups, self.hidden_channels)
        # for _ in range(self.time_step_num):
        #     self.attention.append(LowRankAttention(self.k, self.hidden_channels, self.dropout))
        #     self.dimension_reduce.append(Sequential(Linear(2*self.k + self.hidden_channels, self.hidden_channels)))

        # initialize layer parameters
        if self.with_cleanclf is False:
            self.rel_compress = Linear(self.pooling_dim, self.num_rel_cls)
            self.ctx_compress = Linear(self.hidden_dim * 2, self.num_rel_cls)
            layer_init_kaiming_normal(self.rel_compress)
            layer_init_kaiming_normal(self.ctx_compress)
            if self.use_bias:
                # convey statistics into FrequencyBias to avoid loading again
                self.freq_bias = FrequencyBias(config, statistics)

        # the transfer classifier
        if self.with_cleanclf:
            self.rel_compress_clean = Linear(self.pooling_dim, self.num_rel_cls)
            self.ctx_compress_clean = Linear(self.hidden_dim * 2, self.num_rel_cls)
            layer_init_kaiming_normal(self.rel_compress_clean)
            layer_init_kaiming_normal(self.ctx_compress_clean)
            # self.gcns_rel_clean = GCN(self.pooling_dim, self.pooling_dim, self.dropout_rate)
            # self.gcns_ctx_clean = GCN(self.hidden_dim * 2, self.hidden_dim * 2, self.dropout_rate)
            if self.use_bias:
                self.freq_bias_clean = FrequencyBias(config, statistics)
        if self.with_transfer:
            #pred_adj_np = np.load('./misc/conf_mat_adj_mat.npy')
            print("Using Confusion Matrix Transfer!")
            pred_adj_np = np.load('./misc/conf_mat_freq_train.npy')
            # pred_adj_np = 1.0 - pred_adj_np
            pred_adj_np[0, :] = 0.0
            pred_adj_np[:, 0] = 0.0
            pred_adj_np[0, 0] = 1.0
            # adj_i_j means the baseline outputs category j, but the ground truth is i.
            pred_adj_np = pred_adj_np / (pred_adj_np.sum(-1)[:, None] + 1e-8)
            pred_adj_np = adj_normalize(pred_adj_np)
            self.pred_adj_nor = torch_as_tensor(pred_adj_np, dtype=torch_float32).to(self.devices)
            # self.pred_adj_layer_clean = Linear(self.num_rel_cls, self.num_rel_cls, bias=False)
            # #layer_init(self.pred_adj_layer_clean, xavier=True)
            # with torch_no_grad():
            #     self.pred_adj_layer_clean.weight.copy_(torch.eye(self.num_rel_cls,dtype=torch.float), non_blocking=True)
                #self.pred_adj_layer_clean.weight.copy_(self.pred_adj_nor, non_blocking=True)

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
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        # (Pdb) edge_ctx.size()
        # torch.Size([1280, 768])
        edge_rep = self.post_emb(edge_ctx)
        del edge_ctx
        # (Pdb) edge_ctx.size()
        # torch.Size([1280, 1536])
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim) # torch.Size([1280, 2, 768])

        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim) # torch.Size([1280, 768])
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim) # torch.Size([1280, 768])
        del edge_rep

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0) # 16 * torch.Size([80, 768])
        tail_reps = tail_rep.split(num_objs, dim=0) # 16 * torch.Size([80, 768])
        obj_preds = obj_preds.split(num_objs, dim=0) # 16 * torch.Size([80])
        del head_rep, tail_rep
        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch_cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(obj_pred[pair_idx])
        del global_image_features, head_reps, tail_reps, obj_preds
        prod_rep = cat(prod_reps, dim=0) # torch.Size([3009, 1536]) torch.Size([5022, 1536]) # # REVIEW: Is this some sort of stateful bug?
        del prod_reps
        pair_pred = cat(pair_preds, dim=0) # torch.Size([3009, 2])
        del pair_preds

        # Union Context
        # 1. Just the context suboutput
        # 1A. context
        # union_ctx = self.context_union(union_features, num_rels) # torch.Size([506, 4096]) =>
        # del union_features
        # # 2. Post context for the overall model
        # union_rep = self.post_emb_union(union_ctxself.pooling_dim
        # del union_ctx
        ctx_gate = self.post_cat(prod_rep) # torch.Size([3009, 4096]) # TODO: Use F_sigmoid?
        visual_rep = union_features * ctx_gate
        del ctx_gate

        # First pass
        x_local = F_relu(visual_rep)
        x_local = F_dropout(x_local, p=self.dropout, training=self.training)
        x_global = self.attention_1(union_features)
        del union_features
        x = self.dimension_reduce_1(torch_cat((x_global, x_local), dim=1))
        x = F_relu(x)
        x = self.gn(x)

        # Second/last pass
        x_local = F_relu(x)
        x_local = F_dropout(x_local, p=self.dropout, training=self.training)
        x_global = self.attention_2(x) # TOOD: or union_features?
        del x
        visual_rep = self.dimension_reduce_2(torch_cat((x_global, x_local), dim=1))
        del x_local, x_global

        if not self.with_cleanclf:
            rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep) # TODO: this is new # TODO need to match which of the 3009 belong to which image. Need to up dim but with unravling.
            del visual_rep, prod_rep
            if self.use_bias: # True
                freq_dists_bias = self.freq_bias.index_with_labels(pair_pred)
                del pair_pred
                rel_dists += F_dropout(freq_dists_bias, 0.3, training=self.training) # torch.Size([3009, 51])
        # the transfer classifier
        if self.with_cleanclf:
            rel_dists = self.rel_compress_clean(visual_rep) + self.ctx_compress_clean(prod_rep)
            del visual_rep, prod_rep
            if self.use_bias:
                freq_dists_bias = self.freq_bias_clean.index_with_labels(pair_pred)
                del pair_pred
                rel_dists += F_dropout(freq_dists_bias, 0.3, training=self.training)
        del freq_dists_bias

        if self.with_transfer:
            rel_dists = (self.pred_adj_nor @ rel_dists.T).T

        add_losses = {}
        # if self.with_knowdist:
        #     rel_dists_specific_soft = F.log_softmax(rel_dists, -1)
        #     rel_dists_general_soft = F_softmax(rel_dists_general, -1)
        #     add_losses['know_dist_kl'] = self.kd_alpha * self.kl_loss(rel_dists_specific_soft, rel_dists_general_soft)

        obj_dists = obj_dists.split(num_objs, dim=0) # torch.Size([1280, 151]) => 16 * torch.Size([80, 151])
        rel_dists = rel_dists.split(num_rels, dim=0) # torch.Size([5022, 51]) => (Pdb) rel_dists.split(num_rels, dim=0)[0].size() torch.Size([156, 51]), 240, ...

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        return obj_dists, rel_dists, add_losses


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
