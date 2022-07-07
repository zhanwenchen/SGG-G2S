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
        # TODO: put LRGA in its own class
        self.time_step_num = config.MODEL.ROI_RELATION_HEAD.LRGA.TIME_STEP_NUM
        self.k = config.MODEL.ROI_RELATION_HEAD.LRGA.K
        self.num_groups = config.MODEL.ROI_RELATION_HEAD.LRGA.NUM_GROUPS
        self.dropout = config.MODEL.ROI_RELATION_HEAD.LRGA.DROPOUT
        self.in_channels = self.pooling_dim

        self.use_gsc = config.MODEL.ROI_RELATION_HEAD.USE_GSC
        if self.use_gsc:
            self.gsc_classify_dim_1 = 64
            self.gsc_classify_1 = Linear(self.pooling_dim, self.gsc_classify_dim_1)
            layer_init_kaiming_normal(self.gsc_classify_1)
            self.gsc_classify_dim_2 = 1024
            self.gsc_classify_2 = Linear(self.gsc_classify_dim_1, self.gsc_classify_dim_2)
            layer_init_kaiming_normal(self.gsc_classify_2)
            self.attention_1 = LowRankAttention(self.k, self.in_channels, self.dropout)
            self.dimension_reduce_1 = Sequential(Linear(2*self.k + self.gsc_classify_dim_1, self.gsc_classify_dim_1), ReLU())

            self.gn = GroupNorm(self.num_groups, self.gsc_classify_dim_1)

            self.attention_2 = LowRankAttention(self.k, self.gsc_classify_dim_1, self.dropout)
            self.dimension_reduce_2 = Sequential(Linear(2*self.k + self.gsc_classify_dim_1, self.gsc_classify_dim_2))

            self.gsc_compress = Linear(self.gsc_classify_dim_2, self.num_rel_cls)
            layer_init_kaiming_normal(self.gsc_compress)

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
            layer_init_kaiming_normal(self.rel_compress_clean)
            self.ctx_compress_clean = Linear(self.hidden_dim * 2, self.num_rel_cls)
            layer_init_kaiming_normal(self.ctx_compress_clean)
            if self.use_gsc:
                self.gsc_compress_clean = Linear(self.gsc_classify_dim_2, self.num_rel_cls)
                layer_init_kaiming_normal(self.gsc_compress_clean)
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
        device = union_features.device
        if self.use_gsc:
            pairs_culsum = 0
            new_to_old = torch_empty(union_features.size(0), dtype=int, device=device) # [3298]
        for img_idx, (pair_idx, head_rep, tail_rep, obj_pred) in enumerate(zip(rel_pair_idxs, head_reps, tail_reps, obj_preds)):
            prod_reps.append(torch_cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(obj_pred[pair_idx])
            if self.use_gsc:
                new_to_old[pairs_culsum:pairs_culsum+len(pair_idx)] = img_idx
                pairs_culsum += len(pair_idx)
        del rel_pair_idxs, head_reps, tail_reps, obj_preds, img_idx, pair_idx, head_rep, tail_rep, obj_pred,
        if self.use_gsc:
            del pairs_culsum
        prod_rep = cat(prod_reps, dim=0) # torch.Size([3009, 1536]) torch.Size([5022, 1536]) # # REVIEW: Is this some sort of stateful bug?
        del prod_reps
        pair_pred = cat(pair_preds, dim=0) # torch.Size([3009, 2])
        del pair_preds
        if self.use_gsc:
            global_features_mapped = global_image_features[new_to_old].to(device)
            del global_image_features, new_to_old, device

        if self.use_gsc:
            # GSC Context

            # 1. Just the context suboutput
            # 1A. context
            # num_images = [1 for _ in range(global_features_mapped.size(0))]
            # gsc_ctx = self.context_gsc(global_features_mapped, num_images) # torch.Size([506, 4096]) =>

            # 2. Post context for the overall model
            # gsc_rep = self.post_emb_gsc(gsc_ctx) # [num_unions, 768] => [num_unions, 768]

            # First pass
            # TODO: Which kind of op? Let's start off with a simple linear
            # But linear to what?
            # 1. arbitarily large (like 1024)
            # 2. arbitarily bottlenecked (like 64)
            # 3. same (4096)
            x_local = self.gsc_classify_1(global_features_mapped) # [506, 4096] => [506, 64]
            x_local = F_relu(x_local) # TODO: need some sort of layers for repr learn?
            x_local = F_dropout(x_local, p=self.dropout, training=self.training)
            x_global = self.attention_1(global_features_mapped) # torch.Size([506, 100])
            del global_features_mapped
            x = self.dimension_reduce_1(torch_cat((x_global, x_local), dim=1)) # torch.Size([506, 64])
            x = self.gn(x) # torch.Size([506, 64])

            # Second/last pass
            x_local = self.gsc_classify_2(x)
            x_local = F_relu(x)
            x_local = F_dropout(x_local, p=self.dropout, training=self.training)
            x_global = self.attention_2(x) # TOOD: or union_features?
            del x
            gsc_rep = self.dimension_reduce_2(torch_cat((x_global, x_local), dim=1))
            del x_local, x_global

        ctx_gate = self.post_cat(prod_rep) # torch.Size([3009, 4096]) # TODO: Use F_sigmoid?
        visual_rep = union_features * ctx_gate
        del ctx_gate

        if not self.with_cleanclf:
            rel_dists = self.ctx_compress(prod_rep) + self.rel_compress(visual_rep)  # TODO: this is new # TODO need to match which of the 3009 belong to which image. Need to up dim but with unravling.
            del visual_rep, prod_rep
            if self.use_gsc:
                rel_dists += self.gsc_compress(gsc_rep)
                del gsc_rep
            if self.use_bias: # True
                freq_dists_bias = self.freq_bias.index_with_labels(pair_pred)
                del pair_pred
                rel_dists += F_dropout(freq_dists_bias, 0.3, training=self.training) # torch.Size([3009, 51])
        # the transfer classifier
        if self.with_cleanclf:
            rel_dists = self.ctx_compress_clean(prod_rep) + self.rel_compress_clean(visual_rep)
            del visual_rep, prod_rep
            if self.use_gsc:
                rel_dists += self.gsc_compress_clean(gsc_rep)
                del gsc_rep
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
