# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from torch import cat as torch_cat, stack as torch_stack, no_grad as torch_no_grad, from_numpy as torch_from_numpy, sigmoid as torch_sigmoid, zeros as torch_zeros, arange as torch_arange, empty_like as torch_empty_like, empty as torch_empty, int64 as torch_int64, ones as torch_ones
from torch.nn import Module, Sequential, Linear, ReLU
from torch.nn.functional import dropout as F_dropout, binary_cross_entropy_with_logits as F_binary_cross_entropy_with_logits, relu as F_relu, softmax as F_softmax, cross_entropy as F_cross_entropy
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.layers import Label_Smoothing_Regression
from maskrcnn_benchmark.layers.gcn._utils import adj_normalize
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from .model_transformer_gscnaiveinside import TransformerContextNaiveInside, TransformerEncoder
from .utils_relation import layer_init, get_box_info, get_box_pair_info

#
# @registry.ROI_RELATION_PREDICTOR.register("TransformerTransferPredictor")
# class TransformerTransferPredictor(Module):
#     def __init__(self, config, in_channels):
#         super(TransformerTransferPredictor, self).__init__()
#         self.attribute_on = config.MODEL.ATTRIBUTE_ON
#         # load parameters
#         self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
#         self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
#         self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
#         self.dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
#
#         assert in_channels is not None
#         self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
#         self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
#         self.with_knowdist = False
#         self.devices = config.MODEL.DEVICE
#         self.with_transfer = config.MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER
#         self.with_cleanclf = config.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER
#         # load class dict
#         statistics = get_dataset_statistics(config)
#         obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
#             'att_classes']
#         assert self.num_obj_cls == len(obj_classes)
#         assert self.num_att_cls == len(att_classes)
#         assert self.num_rel_cls == len(rel_classes)
#         self.val_alpha = config.MODEL.ROI_RELATION_HEAD.VAL_ALPHA
#
#         # module construct
#         self.context_roi = TransformerContext(config, obj_classes, rel_classes, in_channels)
#
#         # post decoding
#         self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
#         self.context_pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
#         self.post_emb = Linear(self.hidden_dim, self.hidden_dim * 2)
#         layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
#         self.post_cat = Linear(self.hidden_dim * 2, self.context_pooling_dim)
#         layer_init(self.post_cat, xavier=True)
#
#         if self.context_pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
#             self.union_single_not_match = True
#             self.up_dim = Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.context_pooling_dim)
#             layer_init(self.up_dim, xavier=True)
#         else:
#             self.union_single_not_match = False
#
#         # initialize layer parameters
#         self.rel_compress = Linear(self.context_pooling_dim, self.num_rel_cls)
#         self.ctx_compress = Linear(self.hidden_dim * 2, self.num_rel_cls)
#         layer_init(self.rel_compress, xavier=True)
#         layer_init(self.ctx_compress, xavier=True)
#         if self.use_bias:
#             # convey statistics into FrequencyBias to avoid loading again
#             self.freq_bias = FrequencyBias(config, statistics)
#
#         # the transfer classifier
#         if self.with_cleanclf:
#             self.rel_compress_union_clean = Linear(self.context_pooling_dim, self.num_rel_cls)
#             self.rel_compress_roi_clean = Linear(self.hidden_dim * 2, self.num_rel_cls)
#             layer_init(self.rel_compress_union_clean, xavier=True)
#             layer_init(self.rel_compress_roi_clean, xavier=True)
#             # self.gcns_rel_clean = GCN(self.context_pooling_dim, self.context_pooling_dim, self.dropout_rate)
#             # self.gcns_ctx_clean = GCN(self.hidden_dim * 2, self.hidden_dim * 2, self.dropout_rate)
#             self.freq_bias_clean = FrequencyBias(config, statistics)
#         if self.with_transfer:
#             #pred_adj_np = np.load('./misc/conf_mat_adj_mat.npy')
#             print("Using Confusion Matrix Transfer!")
#             pred_adj_np = np.load('./misc/conf_mat_freq_train.npy')
#             # pred_adj_np = 1.0 - pred_adj_np
#             pred_adj_np[0, :] = 0.0
#             pred_adj_np[:, 0] = 0.0
#             pred_adj_np[0, 0] = 1.0
#             # adj_i_j means the baseline outputs category j, but the ground truth is i.
#             pred_adj_np = pred_adj_np / (pred_adj_np.sum(-1)[:, None] + 1e-8)
#             pred_adj_np = adj_normalize(pred_adj_np)
#             self.pred_adj_nor = torch_from_numpy(pred_adj_np).float().to(self.devices)
#             # self.pred_adj_layer_clean = Linear(self.num_rel_cls, self.num_rel_cls, bias=False)
#             # #layer_init(self.pred_adj_layer_clean, xavier=True)
#             # with torch_no_grad():
#             #     self.pred_adj_layer_clean.weight.copy_(torch.eye(self.num_rel_cls,dtype=torch.float), non_blocking=True)
#                 #self.pred_adj_layer_clean.weight.copy_(self.pred_adj_nor, non_blocking=True)
#     # TODO: make this into a full model right here, with passing subjects and objects, et al.
#     def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, global_image_features=None):
#         """
#         Args:
#             proposals: list[Tensor]: of length batch_size
#                        len(proposals) = 16
#                        proposal_0 = proposals[0]
#                        proposal_0 is BoxList(num_boxes=80, image_width=600, image_height=900, mode=xyxy)
#                        len(proposal_0) = 80
#                        proposal_0[[0]] is the way to access the boxlist
#             obj_dists (list[Tensor]): logits of object label distribution
#             rel_dists (list[Tensor])
#             rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
#                                           (Pdb) rel_pair_idxs[0].size()
#                                           torch.Size([156, 2])
#
#             union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
#                                      (Pdb) union_features.size()
#                                      torch.Size([3009, 4096])
#             global_image_features: torch.Size([16, 4096]
#         Returns:
#             obj_dists (list[Tensor]): logits of object label distribution
#             rel_dists (list[Tensor])
#             rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
#             union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
#         """
#         if self.attribute_on:
#             obj_dists, obj_preds, att_dists, edge_ctx = self.context_roi(roi_features, proposals, logger)
#         else:
#             obj_dists, obj_preds, edge_ctx = self.context_roi(roi_features, proposals, logger)
#
#         # post decode
#         # (Pdb) edge_ctx.size()
#         # torch.Size([1280, 768])
#         edge_rep = self.post_emb(edge_ctx)
#         # (Pdb) edge_ctx.size()
#         # torch.Size([1280, 1536])
#         edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim) # torch.Size([1280, 2, 768])
#
#         head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim) # torch.Size([1280, 768])
#         tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim) # torch.Size([1280, 768])
#
#         num_rels = [r.shape[0] for r in rel_pair_idxs]
#         num_objs = [len(b) for b in proposals]
#         assert len(num_rels) == len(num_objs)
#
#         head_reps = head_rep.split(num_objs, dim=0) # 16 * torch.Size([80, 768])
#         tail_reps = tail_rep.split(num_objs, dim=0) # 16 * torch.Size([80, 768])
#         obj_preds = obj_preds.split(num_objs, dim=0) # 16 * torch.Size([80])
#
#         # from object level feature to pairwise relation level feature
#         prod_reps = []
#         pair_preds = []
#         pairs_culsum = 0
#         global_features_mapped = torch_empty_like(union_features)
#         # new_to_old = torch_empty(union_features.size(0), dtype=int)
#         for img_idx, (pair_idx, head_rep, tail_rep, obj_pred) in enumerate(zip(rel_pair_idxs, head_reps, tail_reps, obj_preds)):
#             prod_reps.append(torch_cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
#             pair_preds.append(torch_stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
#             # new_to_old[pairs_culsum:pairs_culsum+len(pair_idx)] = img_idx
#             global_features_mapped[pairs_culsum:pairs_culsum+len(pair_idx)] = global_image_features[img_idx]
#             pairs_culsum += len(pair_idx)
#         del global_image_features
#         prod_rep = cat(prod_reps, dim=0) # torch.Size([3009, 1536]) torch.Size([5022, 1536]) # # REVIEW: Is this some sort of stateful bug?
#         pair_pred = cat(pair_preds, dim=0) # torch.Size([3009, 2])
#         # global_features_mapped[new_to_old] = global_image_features
#
#         ctx_gate = self.post_cat(prod_rep) # torch.Size([3009, 4096])
#
#         # use union box and mask convolution
#         if self.use_vision: # True
#             if self.union_single_not_match: # False
#                 visual_rep = ctx_gate * self.up_dim(union_features)
#             else:
#                 visual_rep = ctx_gate * union_features # torch.Size([3009, 4096])
#
#         # rel_dists_general = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep) # A: bottlenecked edge * union features; => rels B: bottlenecked edge. torch.Size([3009, 51]) for all 3
#         rel_dists_general = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep) # TODO: this is new # TODO need to match which of the 3009 belong to which image. Need to up dim but with unravling.
#         if self.use_bias: # True
#             freq_dists_bias = self.freq_bias.index_with_labels(pair_pred)
#             freq_dists_bias = F_dropout(freq_dists_bias, 0.3, training=self.training)
#             rel_dists_general = rel_dists_general + freq_dists_bias # torch.Size([3009, 51])
#         rel_dists = rel_dists_general
#         # the transfer classifier
#         if self.with_cleanclf:
#             rel_dists_clean = self.rel_compress_union_clean(visual_rep) + self.rel_compress_roi_clean(prod_rep)
#             if self.use_bias:
#                 freq_dists_bias_clean = self.freq_bias_clean.index_with_labels(pair_pred)
#                 freq_dists_bias_clean = F_dropout(freq_dists_bias_clean, 0.3, training=self.training)
#                 rel_dists_clean = rel_dists_clean + freq_dists_bias_clean
#             rel_dists = rel_dists_clean
#
#         if self.with_transfer:
#             rel_dists = (self.pred_adj_nor @ rel_dists.T).T
#
#         add_losses = {}
#         # if self.with_knowdist:
#         #     rel_dists_specific_soft = F.log_softmax(rel_dists, -1)
#         #     rel_dists_general_soft = F_softmax(rel_dists_general, -1)
#         #     add_losses['know_dist_kl'] = self.kd_alpha * self.kl_loss(rel_dists_specific_soft, rel_dists_general_soft)
#
#         obj_dists = obj_dists.split(num_objs, dim=0) # torch.Size([1280, 151]) => 16 * torch.Size([80, 151])
#         rel_dists = rel_dists.split(num_rels, dim=0) # torch.Size([5022, 51]) => (Pdb) rel_dists.split(num_rels, dim=0)[0].size() torch.Size([156, 51]), 240, ...
#
#         if self.attribute_on:
#             att_dists = att_dists.split(num_objs, dim=0)
#             return (obj_dists, att_dists), rel_dists, add_losses
#         return obj_dists, rel_dists, add_losses
#

@registry.ROI_RELATION_PREDICTOR.register("TransformerTransferGSCNaiveInsidePredictor")
class TransformerTransferGSCNaiveInsidePredictor(Module):
    def __init__(self, config, in_channels):
        super(TransformerTransferGSCNaiveInsidePredictor, self).__init__()
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

        self.obj_classes = obj_classes
        self.rel_classes = rel_classes

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.context_pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.obj_layer = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER
        self.edge_layer = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        self.num_head = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # module construct
        self.union_embed = Linear(self.context_pooling_dim, self.hidden_dim)
        self.gsc_embed = Linear(self.context_pooling_dim, self.hidden_dim)
        self.lin_union = Linear(self.context_pooling_dim + self.hidden_dim, self.hidden_dim)
        self.lin_gsc = Linear(self.context_pooling_dim + self.hidden_dim, self.hidden_dim)

        self.context_roi = TransformerContextNaiveInside(config, obj_classes, rel_classes, in_channels)
        self.context_edge = TransformerEncoder(self.edge_layer, self.num_head, self.k_dim,
                                               self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)

        self.context_union = TransformerEncoder(self.obj_layer, self.num_head, self.k_dim,
                                               self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)

        self.context_gsc = TransformerEncoder(self.obj_layer, self.num_head, self.k_dim,
                                               self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)

        # post decoding
        self.post_emb = Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_emb_union = Linear(self.hidden_dim, self.hidden_dim)
        self.post_emb_gsc = Linear(self.hidden_dim, self.hidden_dim)
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_emb_union, 10.0 * (1.0 / self.hidden_dim), normal=True)
        layer_init(self.post_emb_gsc, 10.0 * (1.0 / self.hidden_dim), normal=True)
        # self.post_cat = Linear(self.hidden_dim * 2, self.context_pooling_dim) # used to be 1536, 4096
        self.post_cat = Linear(self.hidden_dim * 2, self.hidden_dim,)
        layer_init(self.post_cat, xavier=True)

        if self.context_pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.context_pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # initialize layer parameters
        self.rel_compress_roi = Linear(self.hidden_dim * 2, self.num_rel_cls)
        self.rel_compress_union = Linear(self.hidden_dim, self.num_rel_cls)
        self.rel_compress_gsc = Linear(self.hidden_dim, self.num_rel_cls)
        layer_init(self.rel_compress_roi, xavier=True)
        layer_init(self.rel_compress_union, xavier=True)
        layer_init(self.rel_compress_gsc, xavier=True)
        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        # the transfer classifier
        if self.with_cleanclf:
            self.rel_compress_roi_clean = Linear(self.hidden_dim * 2, self.num_rel_cls)
            self.rel_compress_union_clean = Linear(self.context_pooling_dim, self.num_rel_cls)
            self.rel_compress_gsc_clean = Linear(self.context_pooling_dim, self.num_rel_cls)
            layer_init(self.rel_compress_roi_clean, xavier=True)
            layer_init(self.rel_compress_union_clean, xavier=True)
            layer_init(self.rel_compress_gsc_clean, xavier=True)
            # self.gcns_rel_clean = GCN(self.context_pooling_dim, self.context_pooling_dim, self.dropout_rate)
            # self.gcns_ctx_clean = GCN(self.hidden_dim * 2, self.hidden_dim * 2, self.dropout_rate)
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
            self.pred_adj_nor = torch_from_numpy(pred_adj_np).float().to(self.devices)
            # self.pred_adj_layer_clean = Linear(self.num_rel_cls, self.num_rel_cls, bias=False)
            # #layer_init(self.pred_adj_layer_clean, xavier=True)
            # with torch_no_grad():
            #     self.pred_adj_layer_clean.weight.copy_(torch.eye(self.num_rel_cls,dtype=torch.float), non_blocking=True)
                #self.pred_adj_layer_clean.weight.copy_(self.pred_adj_nor, non_blocking=True)

    # TODO: make this into a full model right here, with passing subjects and objects, et al.
    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, global_image_features=None, logger=None):
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
            global_image_features: torch.Size([16, 4096]
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_roi(roi_features, proposals, logger=logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_roi(roi_features, proposals, logger=logger)

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

        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0) # 16 * torch.Size([80, 768])
        del head_rep
        tail_reps = tail_rep.split(num_objs, dim=0) # 16 * torch.Size([80, 768])
        del tail_rep
        obj_preds = obj_preds.split(num_objs, dim=0) # 16 * torch.Size([80])

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        relation_image_indices = []
        # relation_image_indices = torch_ones(, device=global_features.device, dtype=torch_int64, requires_grad=False) # torch.Size([746, 4096]). global_image_features = torch.Size([16, 4096]
        # global_features_mapped = torch_empty_like(union_features, device=union_features.device, dtype=union_features.dtype) # torch.Size([746, 4096]). global_image_features = torch.Size([16, 4096]
        # new_to_old = torch_empty(union_features.size(0), dtype=int)
        # Iterate by each image in the batch
        # TODO: opportunity to cat lists first and then vectorize
        for img_idx, (pair_idx, head_rep, tail_rep, obj_pred) in enumerate(zip(rel_pair_idxs, head_reps, tail_reps, obj_preds)):
            head_idx = pair_idx[:, 0] # 506
            tail_idx = pair_idx[:, 1] # 506
            prod_reps.append(torch_cat((head_rep[head_idx], tail_rep[tail_idx]), dim=-1)) # torch.Size([156, 1536])
            pair_preds.append(obj_pred[pair_idx])
            # pair_preds.append(torch_stack((obj_pred[head_idx], obj_pred[tail_idx]), dim=1)) # torch.Size([156, 2])
            # new_to_old[pairs_culsum:pairs_culsum+len(pair_idx)] = img_idx
            # relation_image_indices[pairs_culsum:pairs_culsum+len(pair_idx)] = img_idx
            relation_image_indices.extend([img_idx for _ in range(len(pair_idx))]) # [16, any]: [[1, 1, 1, 1, 1], [2, 2,], [3, 3, 3]]
        del head_idx, tail_idx, head_reps, tail_reps, obj_preds, rel_pair_idxs, head_rep, tail_rep

            # global_features_mapped[pairs_culsum:pairs_culsum+len(pair_idx)] = global_image_features[img_idx]
        # del global_image_features
        prod_rep = cat(prod_reps, dim=0) # torch.Size([3009, 1536]) torch.Size([5022, 1536]) # # REVIEW: Is this some sort of stateful bug? No. Just shuffling during training
        del prod_reps

        pair_pred = cat(pair_preds, dim=0) # torch.Size([3009, 2])
        del pair_preds
        # global_features_mapped[new_to_old] = global_image_features
        # Does global_image_features require grad?
        global_features_mapped = global_image_features[relation_image_indices] # torch.Size([746, 4096])
        del global_image_features, relation_image_indices
        ctx_gate = self.post_cat(prod_rep) # Old: torch.Size([3009, 4096]). New: torch.Size([746, 768])
        # TODO: design choices: do we use the roi union ctx_gate for union features and gsc? Tang did for union_features.
        # We need to

        # rel_dists_general = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep) # A: bottlenecked edge * union features; => rels B: bottlenecked edge. torch.Size([3009, 51]) for all 3
        union_pre_rep = cat((union_features, self.union_embed(union_features)), dim=-1) # New: torch.Size([746, 4864]). Old: torch.Size([746, 4096]) + torch.Size([746, 768]) => torch.Size([746, 4864])
        union_ctx = self.context_union(self.lin_union(union_pre_rep), num_rels) # New: torch.Size([746, 768]) # lin_union: torch.Size([746, 4864])=>torch.Size([746, 768]) # sum(num_objs) need to equal 746
        del union_pre_rep
        union_post_ctx = self.post_emb_union(union_ctx) # torch.Size([3009, 768])
        del union_ctx
        union_rep = ctx_gate * union_post_ctx # torch.Size([3009, 4096])
        del union_post_ctx

        # Generate gsc_rep
        gsc_pre_rep = cat((global_features_mapped, self.gsc_embed(global_features_mapped)), dim=-1) # torch.Size([3009, 4096]) + torch.Size([3009, 768]) => # This is where it is if we want to include global features in the edge computation
        num_images = [1 for _ in range(global_features_mapped.size(0))]
        del global_features_mapped
        gsc_ctx = self.context_gsc(self.lin_gsc(gsc_pre_rep), num_images)
        del num_images, gsc_pre_rep
        gsc_post_ctx = self.post_emb_gsc(gsc_ctx) # torch.Size([3009, 768])
        del gsc_ctx
        gsc_rep = ctx_gate * gsc_post_ctx
        del ctx_gate, gsc_post_ctx

        # the transfer classifier
        if self.with_cleanclf:
            rel_dists = self.rel_compress_union_clean(union_rep) + self.rel_compress_roi_clean(prod_rep) + self.rel_compress_gsc_clean(gsc_rep)
            if self.use_bias:
                freq_dists_bias = self.freq_bias_clean.index_with_labels(pair_pred)
                freq_dists_bias = F_dropout(freq_dists_bias, 0.3, training=self.training)
                rel_dists += freq_dists_bias
        else:
            rel_dists = self.rel_compress_union(union_rep) + self.rel_compress_roi(prod_rep) + self.rel_compress_gsc(gsc_rep) # torch.Size([746, 768]) + torch.Size([746, 51])
            if self.use_bias: # True
                freq_dists_bias = self.freq_bias.index_with_labels(pair_pred)
                freq_dists_bias = F_dropout(freq_dists_bias, 0.3, training=self.training)
                rel_dists += freq_dists_bias # torch.Size([3009, 51])

        del prod_rep, union_rep, gsc_rep, freq_dists_bias
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
