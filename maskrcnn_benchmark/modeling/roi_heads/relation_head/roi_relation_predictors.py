# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from torch import (
    cat as torch_cat,
    as_tensor as torch_as_tensor,
    float32 as torch_float32,
    stack as torch_stack,
    from_numpy as torch_from_numpy,
)
# from torch.nn import Module, Linear, ModuleList, Sequential, GroupNorm, ReLU, BatchNorm2d
from torch.nn import Module, Linear, ReLU, BatchNorm1d, TransformerEncoderLayer, TransformerEncoder, Sequential, GroupNorm
from torch.nn.functional import dropout as F_dropout, relu as F_relu, binary_cross_entropy_with_logits as F_binary_cross_entropy_with_logits
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.layers.gcn._utils import adj_normalize
from .model_motifs import LSTMContext, FrequencyBias, SpuerFrequencyBias
from .model_transformer import TransformerContext
from .model_vctree import VCTreeLSTMContext
from .utils_relation import layer_init_kaiming_normal, layer_init
from .axial_attention import AxialAttention
from .lrga import LowRankAttention


METHODS_DATA_2D = {'concat', 'raw_obj_pairwise'}
METHODS_DATA_1D = {'hadamard', 'mm', 'cosine_similarity'}


@registry.ROI_RELATION_PREDICTOR.register("PairwisePredictor")
class PairwisePredictor(Module):
    def __init__(self, config, in_channels):
        super(PairwisePredictor, self).__init__()
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
        self.devices = devices = config.MODEL.DEVICE
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
        self.hidden_dim = hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
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
        # self.time_step_num = config.MODEL.ROI_RELATION_HEAD.LRGA.TIME_STEP_NUM
        # self.k = config.MODEL.ROI_RELATION_HEAD.LRGA.K
        # self.num_groups = config.MODEL.ROI_RELATION_HEAD.LRGA.NUM_GROUPS
        # self.dropout = config.MODEL.ROI_RELATION_HEAD.LRGA.DROPOUT
        # self.in_channels = self.pooling_dim

        # self.use_gsc = config.MODEL.ROI_RELATION_HEAD.USE_GSC
        # if self.use_gsc:
        #     self.gsc_classify_dim_1 = 64
        #     self.gsc_classify_1 = Linear(self.pooling_dim, self.gsc_classify_dim_1)
        #     layer_init_kaiming_normal(self.gsc_classify_1)
        #     self.gsc_classify_dim_2 = 1024
        #     self.gsc_classify_2 = Linear(self.gsc_classify_dim_1, self.gsc_classify_dim_2)
        #     layer_init_kaiming_normal(self.gsc_classify_2)
        #     self.attention_1 = LowRankAttention(self.k, self.in_channels, self.dropout)
        #     self.dimension_reduce_1 = Sequential(Linear(2*self.k + self.gsc_classify_dim_1, self.gsc_classify_dim_1), ReLU())
        #
        #     self.gn = GroupNorm(self.num_groups, self.gsc_classify_dim_1)
        #
        #     self.attention_2 = LowRankAttention(self.k, self.gsc_classify_dim_1, self.dropout)
        #     self.dimension_reduce_2 = Sequential(Linear(2*self.k + self.gsc_classify_dim_1, self.gsc_classify_dim_2))
        #
        #     self.gsc_compress = Linear(self.gsc_classify_dim_2, self.num_rel_cls)
        #     layer_init_kaiming_normal(self.gsc_compress)

        # initialize layer parameters
        if self.with_cleanclf is False:
            self.rel_compress = Linear(self.pooling_dim, self.num_rel_cls, device=devices)
            # self.ctx_compress = Linear(self.hidden_dim * 2, self.num_rel_cls)
            self.ctx_compress = Linear(self.hidden_dim, self.num_rel_cls, device=devices)
            layer_init_kaiming_normal(self.rel_compress)
            layer_init_kaiming_normal(self.ctx_compress)
            if self.use_bias:
                # convey statistics into FrequencyBias to avoid loading again
                self.freq_bias = FrequencyBias(config, statistics)

        # the transfer classifier
        if self.with_cleanclf:
            self.rel_compress_clean = Linear(self.pooling_dim, self.num_rel_cls)
            layer_init_kaiming_normal(self.rel_compress_clean)
            self.ctx_compress_clean = Linear(self.hidden_dim, self.num_rel_cls)
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

        # f_nn_type = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.F_NN_TYPE

        self.pairwise_method_data = pairwise_method_data = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.PAIRWISE_METHOD_DATA
        self.use_pairwise_l2 = use_pairwise_l2 = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.USE_PAIRWISE_L2
        if use_pairwise_l2 is True:
            self.act_pairwise_obj_ctx = ReLU()
            # self.bn_pairwise_obj_ctx = BatchNorm2d(self.hidden_dim, device=self.devices) # TODO: BatchNorm3d?
            # self.bn_pairwise_obj_ctx = BatchNorm2d(self.hidden_dim, device=self.devices) # TODO: BatchNorm3d?
            self.bn_pairwise_obj_ctx = BatchNorm1d(hidden_dim, device=devices) # TODO: BatchNorm3d?
            # self.f_ab_obj_ctx = Linear(1, 1, device=device)
            # layer_init_kaiming_normal(self.f_ab_obj_ctx)
            assert pairwise_method_data in METHODS_DATA_1D | METHODS_DATA_2D

        self.pairwise_method_func = pairwise_method_func = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.PAIRWISE_METHOD_FUNC
        if pairwise_method_func == 'axial_attention':
            assert pairwise_method_data not in METHODS_DATA_1D
            self.f_ab_obj_ctx = AxialAttention(
                dim=hidden_dim,               # embedding dimension
                dim_index = 2,         # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads = 8,             # number of heads for multi-head attention
                num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
        elif pairwise_method_func == 'mha':
            assert pairwise_method_data in METHODS_DATA_1D
            num_head_pairwise = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.MHA.NUM_HEAD
            num_layers_pairwise = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.MHA.NUM_LAYERS
            self.f_ab_obj_ctx = TransformerEncoder(TransformerEncoderLayer(d_model=hidden_dim, nhead=num_head_pairwise), num_layers_pairwise)
        elif pairwise_method_func == 'lrga':
            # self.time_step_num = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.LRGA.TIME_STEP_NUM
            k = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.LRGA.K
            # self.num_groups = config.MODEL.ROI_RELATION_HEAD.LRGA.NUM_GROUPS
            dropout = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.LRGA.DROPOUT
            # self.gsc_classify_dim_1 = 64
            # self.gsc_classify_1 = Linear(hidden_dim, self.gsc_classify_dim_1)
            # layer_init_kaiming_normal(self.gsc_classify_1)
            # self.gsc_classify_dim_2 = 1024
            # self.gsc_classify_2 = Linear(self.gsc_classify_dim_1, self.gsc_classify_dim_2)
            # layer_init_kaiming_normal(self.gsc_classify_2)
            self.attention_1 = LowRankAttention(k, hidden_dim, dropout)
            self.dimension_reduce_1 = Sequential(Linear(2*k + hidden_dim, hidden_dim, device=devices), ReLU())

            self.bn = BatchNorm1d(hidden_dim, device=devices)
            # self.gn = GroupNorm(self.num_groups, hidden_dim)

            # self.attention_2 = LowRankAttention(self.k, hidden_dim, self.dropout)
            # self.dimension_reduce_2 = Sequential(Linear(2*self.k + hidden_dim, hidden_dim, device=devices), ReLU())

            # self.gsc_compress = Linear(self.gsc_classify_dim_2, self.num_rel_cls)
            # layer_init_kaiming_normal(self.gsc_compress)
        else:
            raise ValueError('')


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
        hidden_dim = self.hidden_dim
        edge_rep = edge_rep.view(edge_rep.size(0), 2, hidden_dim) # torch.Size([1280, 2, 768])

        # head_rep = edge_rep[:, 0].contiguous().view(-1, hidden_dim) # torch.Size([1280, 768])
        # tail_rep = edge_rep[:, 1].contiguous().view(-1, hidden_dim) # torch.Size([1280, 768])
        head_rep = edge_rep[:, 0] # torch.Size([1280, 768])
        tail_rep = edge_rep[:, 1] # torch.Size([1280, 768])
        del edge_rep

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        rel_pair_idxs_global = [] # TODO: construct and fill instead?
        num_objs_culsum = 0
        # TODO: maybe use cumsum as an optimization?
        for rel_pair_idx, num_obj in zip(rel_pair_idxs, num_objs):
            rel_pair_idxs_global.append(rel_pair_idx + num_objs_culsum if num_objs_culsum > 0 else rel_pair_idx)
            num_objs_culsum += num_obj

        rel_pair_idxs_global = torch_cat(rel_pair_idxs_global)
        rel_pair_idxs_global_head = rel_pair_idxs_global[:, 0]
        rel_pair_idxs_global_tail = rel_pair_idxs_global[:, 1]

        prod_rep = torch_cat((head_rep[rel_pair_idxs_global_head], tail_rep[rel_pair_idxs_global_tail]), dim=-1)
        del rel_pair_idxs_global_head, rel_pair_idxs_global_tail
        pair_pred = obj_preds[rel_pair_idxs_global]
        del rel_pair_idxs_global

        # head_reps = head_rep.split(num_objs, dim=0) # 16 * torch.Size([80, 768])
        # tail_reps = tail_rep.split(num_objs, dim=0) # 16 * torch.Size([80, 768])
        # obj_preds = obj_preds.split(num_objs, dim=0) # 16 * torch.Size([80])
        # del head_rep, tail_rep
        # from object level feature to pairwise relation level feature
        # prod_reps = []
        # pair_preds = []
        # device = union_features.device
        # if self.use_gsc:
        #     pairs_culsum = 0
        #     new_to_old = torch_empty(union_features.size(0), dtype=int, device=device) # [3298]



        # for img_idx, (pair_idx, head_rep, tail_rep, obj_pred) in enumerate(zip(rel_pair_idxs, head_reps, tail_reps, obj_preds)):
        #     prod_reps.append(torch_cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
        #     pair_preds.append(obj_pred[pair_idx])
        # print('after prod_reps, pair_preds')
            # if self.use_gsc:
            #     new_to_old[pairs_culsum:pairs_culsum+len(pair_idx)] = img_idx
            #     pairs_culsum += len(pair_idx)
        # del rel_pair_idxs, head_reps, tail_reps, obj_preds, img_idx, pair_idx, head_rep, tail_rep, obj_pred,
        # del rel_pair_idxs, head_reps, tail_reps, obj_preds, img_idx, pair_idx, obj_pred,
        # del rel_pair_idxs, head_reps, tail_reps, obj_preds
        del rel_pair_idxs, obj_preds, head_rep, tail_rep
        # if self.use_gsc:
        #     del pairs_culsum
        # prod_rep = cat(prod_reps, dim=0) # torch.Size([3009, 1536]) torch.Size([5022, 1536]) # # REVIEW: Is this some sort of stateful bug?
        # assert equal(prod_rep_new, prod_rep)
        # del prod_reps
        # pair_pred = cat(pair_preds, dim=0) # torch.Size([3009, 2])
        # assert equal(pair_pred_new, pair_pred)
        # del pair_preds
        # breakpoint()
        # if self.use_gsc:
        #     global_features_mapped = global_image_features[new_to_old].to(device)
        #     del global_image_features, new_to_old, device
        #
        # if self.use_gsc:
        #     # GSC Context
        #
        #     # 1. Just the context suboutput
        #     # 1A. context
        #     # num_images = [1 for _ in range(global_features_mapped.size(0))]
        #     # gsc_ctx = self.context_gsc(global_features_mapped, num_images) # torch.Size([506, 4096]) =>
        #
        #     # 2. Post context for the overall model
        #     # gsc_rep = self.post_emb_gsc(gsc_ctx) # [num_unions, 768] => [num_unions, 768]
        #
        #     # First pass
        #     # TODO: Which kind of op? Let's start off with a simple linear
        #     # But linear to what?
        #     # 1. arbitarily large (like 1024)
        #     # 2. arbitarily bottlenecked (like 64)
        #     # 3. same (4096)
        #     x_local = self.gsc_classify_1(global_features_mapped) # [506, 4096] => [506, 64]
        #     x_local = F_relu(x_local) # TODO: need some sort of layers for repr learn?
        #     x_local = F_dropout(x_local, p=self.dropout, training=self.training)
        #     x_global = self.attention_1(global_features_mapped) # torch.Size([506, 100])
        #     del global_features_mapped
        #     x = self.dimension_reduce_1(torch_cat((x_global, x_local), dim=1)) # torch.Size([506, 64])
        #     x = self.gn(x) # torch.Size([506, 64])
        #
        #     # Second/last pass
        #     x_local = self.gsc_classify_2(x)
        #     x_local = F_relu(x)
        #     x_local = F_dropout(x_local, p=self.dropout, training=self.training)
        #     x_global = self.attention_2(x) # TOOD: or union_features?
        #     del x
        #     gsc_rep = self.dimension_reduce_2(torch_cat((x_global, x_local), dim=1))
        #     del x_local, x_global
        # obj_ctx =
        # obj_ctx_subj = obj_ctx[rel_pair_idxs_global_head] # [506, 768]
        # obj_ctx_obj = obj_ctx[rel_pair_idxs_global_tail] # [506, 768]
        # del obj_ctx

        # for rel_pair_idx, num_obj in zip(rel_pair_idxs, num_objs):
        #     rel_pair_idxs_global.append(rel_pair_idx + num_objs_culsum if num_objs_culsum > 0 else rel_pair_idx)
        #     num_objs_culsum += num_obj
        pairwise_method_data = self.pairwise_method_data
        pairwise_method_func = self.pairwise_method_func
        if self.use_pairwise_l2 is True:
            # obj_ctx_subj = prod_rep[rel_pair_idxs_global_head] # [506, 768]
            # obj_ctx_obj = prod_rep[rel_pair_idxs_global_tail] # [506, 768]
            # del obj_ctx
            # TODO: optimize the permute to reduce the number of permute operations.
            # pairwise_obj_ctx_old = obj_ctx[rel_pair_idxs_global_head, None, :] * obj_ctx[None, rel_pair_idxs_global_tail, :] # torch.Size([506, 506, 768]) # or use roi_features instead
            # TODO Need to normalize obj_ctx_subj obj_ctx_obj before multiplication because small numbers
            # pairwise_obj_ctx = head_rep.unsqueeze(1) * tail_rep.unsqueeze(0) # torch.Size([506, 506, 768]) # or use roi_features instead
            if pairwise_method_data == 'concat':
                head_rep, tail_rep = prod_rep.hsplit(2)
                # pairwise_obj_ctx = head_rep.unsqueeze(1) * tail_rep.unsqueeze(0) # torch.Size([506, 506, 768]) # or use roi_features instead

                pairwise_obj_ctx = torch_stack((head_rep, tail_rep), dim=2) # torch.Size([506, 768, 2]) # or use roi_features instead
                del head_rep, tail_rep
                # print('after stacking prod_reps, pair_preds pairwise_obj_ctx')
                # breakpoint()
                # [1, 506, 768, 2]
                # breakpoint()
                # breakpoint()
                pairwise_obj_ctx = pairwise_obj_ctx.unsqueeze_(0)

            elif pairwise_method_data == 'hadamard':
                head_rep, tail_rep = prod_rep.hsplit(2)
                pairwise_obj_ctx = head_rep * tail_rep


            if pairwise_method_func == 'axial_attention':
                # print('after unsqueeze')
                # pairwise_obj_ctx = self.f_ab_obj_ctx(pairwise_obj_ctx).squeeze_(0) # torch.Size([1, 506, 2, 768])
                # breakpoint()
                # print(f'pairwise_obj_ctx.size() = {pairwise_obj_ctx.size()}')
                pairwise_obj_ctx = self.f_ab_obj_ctx(pairwise_obj_ctx).squeeze_(0) # torch.Size([506, 2, 768])
                # print('after self.f_ab_obj_ctx')
                pairwise_obj_ctx = self.act_pairwise_obj_ctx(pairwise_obj_ctx)
                # print('after self.act_pairwise_obj_ctx')
                # pairwise_obj_ctx = torch_movedim(pairwise_obj_ctx, 3, 1) # torch.Size([1, 768, 506, 2])
                # pairwise_obj_ctx = self.bn_pairwise_obj_ctx(pairwise_obj_ctx).squeeze_(0) # torch.Size([768, 506, 2])
                # pairwise_obj_ctx = self.bn_pairwise_obj_ctx(pairwise_obj_ctx) # torch.Size([768, 506, 2])
                # breakpoint()
                # pairwise_obj_ctx = self.bn_pairwise_obj_ctx(pairwise_obj_ctx.transpose_(1, 2)) # torch.Size([506, 768, 2])
                pairwise_obj_ctx = self.bn_pairwise_obj_ctx(pairwise_obj_ctx) # torch.Size([506, 768, 2])
                # print('after self.bn_pairwise_obj_ctx')
                # pairwise_obj_ctx = torch_movedim(pairwise_obj_ctx, 0, 2) # torch.Size([768, 506, 2])
                # pairwise_obj_ctx = self.bn_pairwise_obj_ctx(.permute(0, 3, 1, 2)).squeeze(0)
                # pairwise_obj_ctx.transpose_()
                # pairwise_obj_ctx = self.bn_pairwise_obj_ctx(.permute(0, 3, 1, 2)).squeeze(0)
                # breakpoint()
                pairwise_obj_ctx = pairwise_obj_ctx.sum(dim=2) # [506, 768] # TODO: prod?
                # print('after sum')
                # ctx_pairwise_rois = obj_features_subject + obj_features_object + pairwise_obj_ctx.transpose_(0, 1)
            elif pairwise_method_func == 'mha':
                pairwise_obj_ctx = self.f_ab_obj_ctx(pairwise_obj_ctx)
            elif pairwise_method_func == 'lrga':
                # x_local = self.f_ab_obj_ctx(pairwise_obj_ctx)
                # x_local = self.gsc_classify_1(global_features_mapped) # [506, 4096] => [506, 64]
                #     x_local = F_relu(x_local) # TODO: need some sort of layers for repr learn?
                #     x_local = F_dropout(x_local, p=self.dropout, training=self.training)
                x_global = self.attention_1(pairwise_obj_ctx) # torch.Size([506, 100])
                #     del global_features_mapped
                pairwise_obj_ctx = self.dimension_reduce_1(torch_cat((x_global, pairwise_obj_ctx), dim=1)) # torch.Size([506, 64])
                pairwise_obj_ctx = self.bn(pairwise_obj_ctx) # torch.Size([506, 64])
#         self.bn = ModuleList([BatchNorm1d(hidden_channels) for _ in range(num_layers-1)])
                #
                #     # Second/last pass
                # x_local = self.gsc_classify_2(pairwise_obj_ctx)
                # x_local = F_relu(pairwise_obj_ctx)
                # x_local = F_dropout(x_local, p=self.dropout, training=self.training)
                # x_global = self.attention_2(pairwise_obj_ctx) # TOOD: or union_features?
                # pairwise_obj_ctx = self.dimension_reduce_2(torch_cat((x_global, pairwise_obj_ctx), dim=1))
                # del x_global
        ctx_gate = self.post_cat(prod_rep) # torch.Size([3009, 4096]) # TODO: Use F_sigmoid?
        # print('after self.post_cat')
        visual_rep = union_features * ctx_gate
        # print('after gating')
        del union_features, ctx_gate

        if not self.with_cleanclf:
            # breakpoint()
            # rel_dists = self.ctx_compress(prod_rep + pairwise_obj_ctx.transpose_(0, 1)) + self.rel_compress(visual_rep)  # TODO: this is new # TODO need to match which of the 3009 belong to which image. Need to up dim but with unravling.
            rel_dists = self.ctx_compress(pairwise_obj_ctx) + self.rel_compress(visual_rep)  # TODO: this is new # TODO need to match which of the 3009 belong to which image. Need to up dim but with unravling.
            # print('after compression')
            del visual_rep, prod_rep
            # if self.use_gsc:
            #     rel_dists += self.gsc_compress(gsc_rep)
            #     del gsc_rep
            if self.use_bias: # True
                freq_dists_bias = self.freq_bias.index_with_labels(pair_pred)
                del pair_pred
                rel_dists += F_dropout(freq_dists_bias, 0.3, training=self.training) # torch.Size([3009, 51])
                # print('after self.use_bias')
        # the transfer classifier
        # if self.with_cleanclf:
        #     rel_dists = self.ctx_compress_clean(prod_rep) + self.rel_compress_clean(visual_rep)
        #     del visual_rep, prod_rep
        #     if self.use_gsc:
        #         rel_dists += self.gsc_compress_clean(gsc_rep)
        #         del gsc_rep
        #     if self.use_bias:
        #         freq_dists_bias = self.freq_bias_clean.index_with_labels(pair_pred)
        #         del pair_pred
        #         rel_dists += F_dropout(freq_dists_bias, 0.3, training=self.training)
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

@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = Linear(self.hidden_dim, self.hidden_dim * 2)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

        self.with_clean_classifier = config.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER

        self.use_pairwise_l2 = use_pairwise_l2 = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.USE_PAIRWISE_L2
        self.pairwise_method_data = pairwise_method_data = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.PAIRWISE_METHOD_DATA
        self.pairwise_method_func = pairwise_method_func = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.PAIRWISE_METHOD_FUNC

        if self.use_pairwise_l2 is True and pairwise_method_func == 'mha':
            assert pairwise_method_data in METHODS_DATA_1D
            num_head_pairwise = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.MHA.NUM_HEAD
            num_layers_pairwise = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.MHA.NUM_LAYERS
            self.f_ab_obj_ctx = TransformerEncoder(TransformerEncoderLayer(d_model=hidden_dim, nhead=num_head_pairwise), num_layers_pairwise)

        if self.with_clean_classifier:
            if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
                self.union_single_not_match = True
                self.up_dim_clean = Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
                layer_init(self.up_dim_clean, xavier=True)
            else:
                self.union_single_not_match = False
            self.post_cat_clean = Linear(self.hidden_dim * 2, self.pooling_dim)
            self.ctx_compress_clean = Linear(self.pooling_dim, self.num_rel_cls, bias=True)
            layer_init(self.post_cat_clean, xavier=True)
            layer_init(self.ctx_compress_clean, xavier=True)

            if self.use_pairwise_l2 is True:
                self.pairwise_compress_clean = Linear(self.hidden_dim, self.num_rel_cls)
                layer_init(self.pairwise_compress_clean, xavier=True)
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias_clean = FrequencyBias(config, statistics)

            self.with_transfer = config.MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER
            if self.with_transfer:
                self.devices = config.MODEL.DEVICE
                print("!!!!!!!!!With Confusion Matrix Channel!!!!!")
                pred_adj_np = np.load('./misc/conf_mat_freq_train.npy')
                # pred_adj_np = 1.0 - pred_adj_np
                pred_adj_np[0, :] = 0.0
                pred_adj_np[:, 0] = 0.0
                pred_adj_np[0, 0] = 1.0
                # adj_i_j means the baseline outputs category j, but the ground truth is i.
                pred_adj_np = pred_adj_np / (pred_adj_np.sum(-1)[:, None] + 1e-8)
                pred_adj_np = adj_normalize(pred_adj_np)
                self.pred_adj_nor = torch_from_numpy(pred_adj_np).float().to(self.devices)
        else:
            # learned-mixin
            # self.uni_gate = Linear(self.pooling_dim, self.num_rel_cls)
            # self.frq_gate = Linear(self.pooling_dim, self.num_rel_cls)
            self.post_cat = Linear(self.hidden_dim * 2, self.pooling_dim)
            layer_init(self.post_cat, xavier=True)

            self.ctx_compress = Linear(self.pooling_dim, self.num_rel_cls)
            # self.uni_compress = Linear(self.pooling_dim, self.num_rel_cls)
            # layer_init(self.uni_gate, xavier=True)
            # layer_init(self.frq_gate, xavier=True)
            layer_init(self.ctx_compress, xavier=True)
            # layer_init(self.uni_compress, xavier=True)

            if self.use_pairwise_l2 is True:
                self.pairwise_compress = Linear(hidden_dim, self.num_rel_cls)
                layer_init(self.pairwise_compress, xavier=True)


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, global_image_features=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs,
                                                                          logger)

        # post decode
        edge_rep = F_relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        # head_reps = head_rep.split(num_objs, dim=0)
        # tail_reps = tail_rep.split(num_objs, dim=0)
        # obj_preds = obj_preds.split(num_objs, dim=0)

        # prod_reps = []
        # pair_preds = []
        # for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
        #     prod_reps.append(torch_cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
        #     pair_preds.append(torch_stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        # prod_rep = torch_cat(prod_reps, dim=0)
        # pair_pred = torch_cat(pair_preds, dim=0)

        rel_pair_idxs_global = [] # TODO: construct and fill instead?
        num_objs_culsum = 0
        # TODO: maybe use cumsum as an optimization?
        for rel_pair_idx, num_obj in zip(rel_pair_idxs, num_objs):
            rel_pair_idxs_global.append(rel_pair_idx + num_objs_culsum if num_objs_culsum > 0 else rel_pair_idx)
            num_objs_culsum += num_obj

        rel_pair_idxs_global = torch_cat(rel_pair_idxs_global)
        rel_pair_idxs_global_head = rel_pair_idxs_global[:, 0]
        rel_pair_idxs_global_tail = rel_pair_idxs_global[:, 1]

        head_rep = head_rep[rel_pair_idxs_global_head]
        tail_rep = tail_rep[rel_pair_idxs_global_tail]
        prod_rep = torch_cat((head_rep, tail_rep), dim=-1)
        del rel_pair_idxs_global_head, rel_pair_idxs_global_tail
        pair_pred = obj_preds[rel_pair_idxs_global]
        del rel_pair_idxs_global

        pairwise_method_data = self.pairwise_method_data
        pairwise_method_func = self.pairwise_method_func
        if self.use_pairwise_l2 is True:
            # obj_ctx_subj = prod_rep[rel_pair_idxs_global_head] # [506, 768]
            # obj_ctx_obj = prod_rep[rel_pair_idxs_global_tail] # [506, 768]
            # del obj_ctx
            # TODO: optimize the permute to reduce the number of permute operations.
            # pairwise_obj_ctx_old = obj_ctx[rel_pair_idxs_global_head, None, :] * obj_ctx[None, rel_pair_idxs_global_tail, :] # torch.Size([506, 506, 768]) # or use roi_features instead
            # TODO Need to normalize obj_ctx_subj obj_ctx_obj before multiplication because small numbers
            # pairwise_obj_ctx = head_rep.unsqueeze(1) * tail_rep.unsqueeze(0) # torch.Size([506, 506, 768]) # or use roi_features instead
            if pairwise_method_data == 'concat':
                head_rep, tail_rep = prod_rep.hsplit(2)
                # pairwise_obj_ctx = head_rep.unsqueeze(1) * tail_rep.unsqueeze(0) # torch.Size([506, 506, 768]) # or use roi_features instead

                pairwise_obj_ctx = torch_stack((head_rep, tail_rep), dim=2) # torch.Size([506, 768, 2]) # or use roi_features instead
                del head_rep, tail_rep
                # print('after stacking prod_reps, pair_preds pairwise_obj_ctx')
                # breakpoint()
                # [1, 506, 768, 2]
                # breakpoint()
                # breakpoint()
                pairwise_obj_ctx = pairwise_obj_ctx.unsqueeze_(0)

            elif pairwise_method_data == 'hadamard':
                pairwise_obj_ctx = head_rep * tail_rep
                del head_rep, tail_rep

            if pairwise_method_func == 'axial_attention':
                # print('after unsqueeze')
                # pairwise_obj_ctx = self.f_ab_obj_ctx(pairwise_obj_ctx).squeeze_(0) # torch.Size([1, 506, 2, 768])
                # breakpoint()
                # print(f'pairwise_obj_ctx.size() = {pairwise_obj_ctx.size()}')
                pairwise_obj_ctx = self.f_ab_obj_ctx(pairwise_obj_ctx).squeeze_(0) # torch.Size([506, 2, 768])
                # print('after self.f_ab_obj_ctx')
                pairwise_obj_ctx = self.act_pairwise_obj_ctx(pairwise_obj_ctx)
                # print('after self.act_pairwise_obj_ctx')
                # pairwise_obj_ctx = torch_movedim(pairwise_obj_ctx, 3, 1) # torch.Size([1, 768, 506, 2])
                # pairwise_obj_ctx = self.bn_pairwise_obj_ctx(pairwise_obj_ctx).squeeze_(0) # torch.Size([768, 506, 2])
                # pairwise_obj_ctx = self.bn_pairwise_obj_ctx(pairwise_obj_ctx) # torch.Size([768, 506, 2])
                # breakpoint()
                # pairwise_obj_ctx = self.bn_pairwise_obj_ctx(pairwise_obj_ctx.transpose_(1, 2)) # torch.Size([506, 768, 2])
                pairwise_obj_ctx = self.bn_pairwise_obj_ctx(pairwise_obj_ctx) # torch.Size([506, 768, 2])
                # print('after self.bn_pairwise_obj_ctx')
                # pairwise_obj_ctx = torch_movedim(pairwise_obj_ctx, 0, 2) # torch.Size([768, 506, 2])
                # pairwise_obj_ctx = self.bn_pairwise_obj_ctx(.permute(0, 3, 1, 2)).squeeze(0)
                # pairwise_obj_ctx.transpose_()
                # pairwise_obj_ctx = self.bn_pairwise_obj_ctx(.permute(0, 3, 1, 2)).squeeze(0)
                # breakpoint()
                pairwise_obj_ctx = pairwise_obj_ctx.sum(dim=2) # [506, 768] # TODO: prod?
                # print('after sum')
                # ctx_pairwise_rois = obj_features_subject + obj_features_object + pairwise_obj_ctx.transpose_(0, 1)
            elif pairwise_method_func == 'mha':
                pairwise_obj_ctx = self.f_ab_obj_ctx(pairwise_obj_ctx)
            elif pairwise_method_func == 'lrga':
                # x_local = self.f_ab_obj_ctx(pairwise_obj_ctx)
                # x_local = self.gsc_classify_1(global_features_mapped) # [506, 4096] => [506, 64]
                #     x_local = F_relu(x_local) # TODO: need some sort of layers for repr learn?
                #     x_local = F_dropout(x_local, p=self.dropout, training=self.training)
                x_global = self.attention_1(pairwise_obj_ctx) # torch.Size([506, 100])
                #     del global_features_mapped
                pairwise_obj_ctx = self.dimension_reduce_1(torch_cat((x_global, pairwise_obj_ctx), dim=1)) # torch.Size([506, 64])
                pairwise_obj_ctx = self.bn(pairwise_obj_ctx) # torch.Size([506, 64])
#         self.bn = ModuleList([BatchNorm1d(hidden_channels) for _ in range(num_layers-1)])
                #
                #     # Second/last pass
                # x_local = self.gsc_classify_2(pairwise_obj_ctx)
                # x_local = F_relu(pairwise_obj_ctx)
                # x_local = F_dropout(x_local, p=self.dropout, training=self.training)
                # x_global = self.attention_2(pairwise_obj_ctx) # TOOD: or union_features?
                # pairwise_obj_ctx = self.dimension_reduce_2(torch_cat((x_global, pairwise_obj_ctx), dim=1))
                # del x_global
        else:
            del head_rep, tail_rep


        # learned-mixin Gate
        # uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        # frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        if self.with_clean_classifier is True:
            prod_rep_clean = self.post_cat_clean(prod_rep)
            if self.union_single_not_match:
                union_features = self.up_dim_clean(union_features)

            if self.use_pairwise_l2 is True:
                ctx_dists_clean = self.pairwise_compress_clean(pairwise_obj_ctx) + self.ctx_compress_clean(prod_rep_clean * union_features)
            else:
                ctx_dists_clean = self.ctx_compress_clean(prod_rep_clean * union_features)
            # uni_dists = self.uni_compress(self.drop(union_features))
            frq_dists_clean = self.freq_bias_clean.index_with_labels(pair_pred.long())
            frq_dists_clean = F_dropout(frq_dists_clean, 0.3, training=self.training)
            rel_dists_clean = ctx_dists_clean + frq_dists_clean
            if self.with_transfer:
                rel_dists_clean = (self.pred_adj_nor @ rel_dists_clean.T).T
            rel_dists = rel_dists_clean
        else:
            prod_rep = self.post_cat(prod_rep)

            if self.use_pairwise_l2 is True:
                ctx_dists = self.pairwise_compress(pairwise_obj_ctx) + self.ctx_compress(prod_rep * union_features)
            else:
                ctx_dists = self.ctx_compress(prod_rep * union_features)
            # uni_dists = self.uni_compress(self.drop(union_features))
            frq_dists = self.freq_bias.index_with_labels(pair_pred.long())
            frq_dists = F_dropout(frq_dists, 0.3, training=self.training)
            rel_dists = ctx_dists + frq_dists
            # rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            if binary_preds[0].requires_grad:
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F_binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.pairwise_method_data = pairwise_method_data = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.PAIRWISE_METHOD_DATA
        self.use_pairwise_l2 = use_pairwise_l2 = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.USE_PAIRWISE_L2
        self.pairwise_method_func = pairwise_method_func = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.PAIRWISE_METHOD_FUNC

        self.post_emb = Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = Linear(self.hidden_dim * 2, self.pooling_dim)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        self.with_clean_classifier = config.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER

        if self.with_clean_classifier:
            if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
                self.union_single_not_match = True
                self.up_dim_clean = Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
                layer_init(self.up_dim_clean, xavier=True)
            else:
                self.union_single_not_match = False
            self.post_cat_clean = Linear(self.hidden_dim * 2, self.pooling_dim)
            layer_init(self.post_cat_clean, xavier=True)
            self.rel_compress_clean = Linear(self.pooling_dim, self.num_rel_cls, bias=True)
            layer_init(self.rel_compress_clean, xavier=True)
            if self.use_pairwise_l2 is True:
                self.pairwise_compress_clean = Linear(self.hidden_dim, self.num_rel_cls, bias=True)
                layer_init(self.pairwise_compress_clean, xavier=True)
            if self.use_bias:
                # convey statistics into FrequencyBias to avoid loading again
                self.freq_bias_clean = FrequencyBias(config, statistics)
            self.with_transfer = config.MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER
            if self.with_transfer:
                self.devices = config.MODEL.DEVICE
                print("!!!!!!!!!With Confusion Matrix Channel!!!!!")
                pred_adj_np = np.load('./misc/conf_mat_freq_train.npy')
                # pred_adj_np = 1.0 - pred_adj_np
                pred_adj_np[0, :] = 0.0
                pred_adj_np[:, 0] = 0.0
                pred_adj_np[0, 0] = 1.0
                # adj_i_j means the baseline outputs category j, but the ground truth is i.
                pred_adj_np = pred_adj_np / (pred_adj_np.sum(-1)[:, None] + 1e-8)
                pred_adj_np = adj_normalize(pred_adj_np)
                self.pred_adj_nor = torch_from_numpy(pred_adj_np).float().to(self.devices)
        else:
            self.rel_compress = Linear(self.pooling_dim, self.num_rel_cls, bias=True)
            layer_init(self.rel_compress, xavier=True)

            if self.use_pairwise_l2 is True:
                self.pairwise_compress = Linear(self.hidden_dim, self.num_rel_cls, bias=True)
                layer_init(self.pairwise_compress, xavier=True)

        if use_pairwise_l2 is True:
            self.act_pairwise_obj_ctx = ReLU()
            # self.bn_pairwise_obj_ctx = BatchNorm2d(self.hidden_dim, device=self.devices) # TODO: BatchNorm3d?
            # self.bn_pairwise_obj_ctx = BatchNorm2d(self.hidden_dim, device=self.devices) # TODO: BatchNorm3d?
            self.bn_pairwise_obj_ctx = BatchNorm1d(hidden_dim) # TODO: BatchNorm3d?
            # self.f_ab_obj_ctx = Linear(1, 1, device=device)
            # layer_init_kaiming_normal(self.f_ab_obj_ctx)
            assert pairwise_method_data in METHODS_DATA_1D | METHODS_DATA_2D

        if pairwise_method_func == 'axial_attention':
            assert pairwise_method_data not in METHODS_DATA_1D
            self.f_ab_obj_ctx = AxialAttention(
                dim=hidden_dim,               # embedding dimension
                dim_index = 2,         # where is the embedding dimension
                # dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                heads = 8,             # number of heads for multi-head attention
                num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
                sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
            )
        elif pairwise_method_func == 'mha':
            assert pairwise_method_data in METHODS_DATA_1D
            num_head_pairwise = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.MHA.NUM_HEAD
            num_layers_pairwise = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.MHA.NUM_LAYERS
            self.f_ab_obj_ctx = TransformerEncoder(TransformerEncoderLayer(d_model=hidden_dim, nhead=num_head_pairwise), num_layers_pairwise)
        elif pairwise_method_func == 'lrga':
            # self.time_step_num = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.LRGA.TIME_STEP_NUM
            k = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.LRGA.K
            # self.num_groups = config.MODEL.ROI_RELATION_HEAD.LRGA.NUM_GROUPS
            dropout = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.LRGA.DROPOUT
            # self.gsc_classify_dim_1 = 64
            # self.gsc_classify_1 = Linear(hidden_dim, self.gsc_classify_dim_1)
            # layer_init_kaiming_normal(self.gsc_classify_1)
            # self.gsc_classify_dim_2 = 1024
            # self.gsc_classify_2 = Linear(self.gsc_classify_dim_1, self.gsc_classify_dim_2)
            # layer_init_kaiming_normal(self.gsc_classify_2)
            self.attention_1 = LowRankAttention(k, hidden_dim, dropout)
            self.dimension_reduce_1 = Sequential(Linear(2*k + hidden_dim, hidden_dim), ReLU())

            self.bn = BatchNorm1d(hidden_dim)
            # self.gn = GroupNorm(self.num_groups, hidden_dim)

            # self.attention_2 = LowRankAttention(self.k, hidden_dim, self.dropout)
            # self.dimension_reduce_2 = Sequential(Linear(2*self.k + hidden_dim, hidden_dim, device=devices), ReLU())

            # self.gsc_compress = Linear(self.gsc_classify_dim_2, self.num_rel_cls)
            # layer_init_kaiming_normal(self.gsc_compress)
        else:
            raise ValueError('')

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, global_image_features=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        rel_pair_idxs_global = [] # TODO: construct and fill instead?
        num_objs_culsum = 0
        # TODO: maybe use cumsum as an optimization?
        for rel_pair_idx, num_obj in zip(rel_pair_idxs, num_objs):
            rel_pair_idxs_global.append(rel_pair_idx + num_objs_culsum if num_objs_culsum > 0 else rel_pair_idx)
            num_objs_culsum += num_obj

        rel_pair_idxs_global = torch_cat(rel_pair_idxs_global)
        rel_pair_idxs_global_head = rel_pair_idxs_global[:, 0]
        rel_pair_idxs_global_tail = rel_pair_idxs_global[:, 1]

        head_rep = head_rep[rel_pair_idxs_global_head]
        tail_rep = tail_rep[rel_pair_idxs_global_tail]
        prod_rep = torch_cat((head_rep, tail_rep), dim=-1)
        del rel_pair_idxs_global_head, rel_pair_idxs_global_tail
        pair_pred = obj_preds[rel_pair_idxs_global]
        del rel_pair_idxs_global

        pairwise_method_data = self.pairwise_method_data
        pairwise_method_func = self.pairwise_method_func
        if self.use_pairwise_l2 is True:
            # obj_ctx_subj = prod_rep[rel_pair_idxs_global_head] # [506, 768]
            # obj_ctx_obj = prod_rep[rel_pair_idxs_global_tail] # [506, 768]
            # del obj_ctx
            # TODO: optimize the permute to reduce the number of permute operations.
            # pairwise_obj_ctx_old = obj_ctx[rel_pair_idxs_global_head, None, :] * obj_ctx[None, rel_pair_idxs_global_tail, :] # torch.Size([506, 506, 768]) # or use roi_features instead
            # TODO Need to normalize obj_ctx_subj obj_ctx_obj before multiplication because small numbers
            # pairwise_obj_ctx = head_rep.unsqueeze(1) * tail_rep.unsqueeze(0) # torch.Size([506, 506, 768]) # or use roi_features instead
            if pairwise_method_data == 'concat':
                head_rep, tail_rep = prod_rep.hsplit(2)
                # pairwise_obj_ctx = head_rep.unsqueeze(1) * tail_rep.unsqueeze(0) # torch.Size([506, 506, 768]) # or use roi_features instead

                pairwise_obj_ctx = torch_stack((head_rep, tail_rep), dim=2) # torch.Size([506, 768, 2]) # or use roi_features instead
                del head_rep, tail_rep
                # print('after stacking prod_reps, pair_preds pairwise_obj_ctx')
                # breakpoint()
                # [1, 506, 768, 2]
                # breakpoint()
                # breakpoint()
                pairwise_obj_ctx = pairwise_obj_ctx.unsqueeze_(0)

            elif pairwise_method_data == 'hadamard':
                pairwise_obj_ctx = head_rep * tail_rep
                del head_rep, tail_rep

            if pairwise_method_func == 'axial_attention':
                # print('after unsqueeze')
                # pairwise_obj_ctx = self.f_ab_obj_ctx(pairwise_obj_ctx).squeeze_(0) # torch.Size([1, 506, 2, 768])
                # breakpoint()
                # print(f'pairwise_obj_ctx.size() = {pairwise_obj_ctx.size()}')
                pairwise_obj_ctx = self.f_ab_obj_ctx(pairwise_obj_ctx).squeeze_(0) # torch.Size([506, 2, 768])
                # print('after self.f_ab_obj_ctx')
                pairwise_obj_ctx = self.act_pairwise_obj_ctx(pairwise_obj_ctx)
                # print('after self.act_pairwise_obj_ctx')
                # pairwise_obj_ctx = torch_movedim(pairwise_obj_ctx, 3, 1) # torch.Size([1, 768, 506, 2])
                # pairwise_obj_ctx = self.bn_pairwise_obj_ctx(pairwise_obj_ctx).squeeze_(0) # torch.Size([768, 506, 2])
                # pairwise_obj_ctx = self.bn_pairwise_obj_ctx(pairwise_obj_ctx) # torch.Size([768, 506, 2])
                # breakpoint()
                # pairwise_obj_ctx = self.bn_pairwise_obj_ctx(pairwise_obj_ctx.transpose_(1, 2)) # torch.Size([506, 768, 2])
                pairwise_obj_ctx = self.bn_pairwise_obj_ctx(pairwise_obj_ctx) # torch.Size([506, 768, 2])
                # print('after self.bn_pairwise_obj_ctx')
                # pairwise_obj_ctx = torch_movedim(pairwise_obj_ctx, 0, 2) # torch.Size([768, 506, 2])
                # pairwise_obj_ctx = self.bn_pairwise_obj_ctx(.permute(0, 3, 1, 2)).squeeze(0)
                # pairwise_obj_ctx.transpose_()
                # pairwise_obj_ctx = self.bn_pairwise_obj_ctx(.permute(0, 3, 1, 2)).squeeze(0)
                # breakpoint()
                pairwise_obj_ctx = pairwise_obj_ctx.sum(dim=2) # [506, 768] # TODO: prod?
                # print('after sum')
                # ctx_pairwise_rois = obj_features_subject + obj_features_object + pairwise_obj_ctx.transpose_(0, 1)
            elif pairwise_method_func == 'mha':
                pairwise_obj_ctx = self.f_ab_obj_ctx(pairwise_obj_ctx)
            elif pairwise_method_func == 'lrga':
                # x_local = self.f_ab_obj_ctx(pairwise_obj_ctx)
                # x_local = self.gsc_classify_1(global_features_mapped) # [506, 4096] => [506, 64]
                #     x_local = F_relu(x_local) # TODO: need some sort of layers for repr learn?
                #     x_local = F_dropout(x_local, p=self.dropout, training=self.training)
                x_global = self.attention_1(pairwise_obj_ctx) # torch.Size([506, 100])
                #     del global_features_mapped
                pairwise_obj_ctx = self.dimension_reduce_1(torch_cat((x_global, pairwise_obj_ctx), dim=1)) # torch.Size([506, 64])
                pairwise_obj_ctx = self.bn(pairwise_obj_ctx) # torch.Size([506, 64])
#         self.bn = ModuleList([BatchNorm1d(hidden_channels) for _ in range(num_layers-1)])
                #
                #     # Second/last pass
                # x_local = self.gsc_classify_2(pairwise_obj_ctx)
                # x_local = F_relu(pairwise_obj_ctx)
                # x_local = F_dropout(x_local, p=self.dropout, training=self.training)
                # x_global = self.attention_2(pairwise_obj_ctx) # TOOD: or union_features?
                # pairwise_obj_ctx = self.dimension_reduce_2(torch_cat((x_global, pairwise_obj_ctx), dim=1))
                # del x_global
        else:
            del head_rep, tail_rep

        if self.with_clean_classifier:
            prod_rep = self.pst_cat_clean(prod_rep)
            if self.use_vision:
                if self.union_single_not_match:
                    union_features = self.up_dim_clean(union_features)

                prod_rep *= union_features

                if self.use_pairwise_l2 is True:
                    rel_dists = self.pairwise_compress_clean(pairwise_obj_ctx) + self.rel_compress_clean(prod_rep)
                    del pairwise_obj_ctx
                else:
                    rel_dists = self.rel_compress_clean(prod_rep)
                del prod_rep
            if self.use_bias:
                freq_dists_bias_clean = self.freq_bias_clean.index_with_labels(pair_pred.long())
                freq_dists_bias_clean = F_dropout(freq_dists_bias_clean, 0.3, training=self.training)
                rel_dists += freq_dists_bias_clean
            if self.with_transfer:
                rel_dists = (self.pred_adj_nor @ rel_dists.T).T
        else:
            prod_rep = self.post_cat(prod_rep)

            if self.use_vision:
                if self.union_single_not_match:
                    union_features = self.up_dim(union_features)

                prod_rep *= union_features

            if self.use_pairwise_l2 is True:
                rel_dists = self.pairwise_compress(pairwise_obj_ctx) + self.rel_compress(prod_rep)
                del pairwise_obj_ctx
            else:
                rel_dists = self.rel_compress(prod_rep)
            del prod_rep
            if self.use_bias:
                freq_dists_bias = self.freq_bias.index_with_labels(pair_pred.long())
                freq_dists_bias = F_dropout(freq_dists_bias, 0.3, training=self.training)
                rel_dists += freq_dists_bias

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
