# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from numpy import unravel_index as np_unravel_index
from torch import (
    cat as torch_cat, no_grad as torch_no_grad,
    as_tensor as torch_as_tensor,
    empty_like as torch_empty_like, empty as torch_empty,
    int64 as torch_int64,
    float32 as torch_float32,
    # abs as torch_abs,
)
from torch.nn import Module, Linear, ReLU, BatchNorm1d, BatchNorm2d, Dropout
from torch.nn.functional import softmax as F_softmax
from torch.cuda import current_device, memory_summary
# from torch.linalg import vector_norm
# from vit_pytorch import ViT
from .axial_attention import AxialAttention
# from transformers import ViTConfig, ViTForImageClassification
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.data import get_dataset_statistics
from .model_motifs import FrequencyBias
# from .model_transformer import TransformerContext, TransformerEncoder
from .utils_relation import layer_init_kaiming_normal, nms_overlaps
from .utils_motifs import to_onehot


@registry.ROI_RELATION_PREDICTOR.register("PairwisePredictor")
class PairwisePredictor(Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        use_gt_box = cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX
        use_gt_object_label = cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        if use_gt_box is True and use_gt_object_label is True:
            self.mode = 'predcls'
        if use_gt_box is True and use_gt_object_label is False:
            self.mode = 'sgcls'
        if use_gt_box is False and use_gt_object_label is False:
            self.mode = 'sgdet'
        if self.mode is None:
            raise ValueError(f'mode is None given use_gt_box={use_gt_box} and use_gt_object_label={use_gt_object_label}')

        self.device = device = current_device()

        self.use_gt_object_label = cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL

        # Data Dimensions
        self.num_obj_cls = num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        # Common nn dims
        hidden_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_cat = Linear(hidden_dim * 2, pooling_dim, device=device)
        layer_init_kaiming_normal(self.post_cat)
        self.out_obj = Linear(hidden_dim, num_obj_cls)
        layer_init_kaiming_normal(self.out_obj)
        self.dropout = Dropout(p=0.3)
        statistics = get_dataset_statistics(cfg)
        self.use_bias = cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(cfg, statistics)

        self.f_obj = Linear(pooling_dim, hidden_dim, device=device)
        layer_init_kaiming_normal(self.f_obj)
        self.act_obj = ReLU()
        self.bn_obj = BatchNorm1d(hidden_dim, device=device) # TODO: BatchNorm3d?

        self.f_a = Linear(hidden_dim, hidden_dim, device=device)
        layer_init_kaiming_normal(self.f_a)
        self.act_obj_subj = ReLU()
        self.bn_obj_subj = BatchNorm1d(hidden_dim, device=device) # TODO: BatchNorm3d?

        self.f_b = Linear(hidden_dim, hidden_dim, device=device)
        layer_init_kaiming_normal(self.f_b)
        self.act_obj_obj = ReLU()
        self.bn_obj_obj = BatchNorm1d(hidden_dim, device=device) # TODO: BatchNorm3d?

        # Pairwise-specific nn dims
        f_nn_type = cfg.MODEL.ROI_RELATION_HEAD.PAIRWISE.F_NN_TYPE
        self.use_pairwise_l1 = use_pairwise_l1 = cfg.MODEL.ROI_RELATION_HEAD.PAIRWISE.USE_PAIRWISE_L1
        self.use_pairwise_l2 = use_pairwise_l2 = cfg.MODEL.ROI_RELATION_HEAD.PAIRWISE.USE_PAIRWISE_L2
        self.use_pairwise_l3 = use_pairwise_l3 = cfg.MODEL.ROI_RELATION_HEAD.PAIRWISE.USE_PAIRWISE_L3
        # pairwise_l1
        # TODO: does batchnorm expect channel first?
        if self.use_pairwise_l1 is True:
            self.act_pairwise_raw = ReLU()
            self.bn_raw = BatchNorm2d(hidden_dim, device=device) # TODO: BatchNorm3d?
        # pairwise l2
        if self.use_pairwise_l2 is True:
            self.act_pairwise_obj_ctx = ReLU()
            self.bn_pairwise_obj_ctx = BatchNorm2d(hidden_dim, device=device) # TODO: BatchNorm3d?
        # pairwise l3
        if self.use_pairwise_l1 is True:
            self.act_pairwise_ctx_subj_obj = ReLU()
            self.bn_pairwise_ctx_subj_obj = BatchNorm2d(hidden_dim, device=device) # TODO: BatchNorm3d?
        # try different kinds of neural networks:
        if f_nn_type == 'linear':
            if use_pairwise_l1 is True:
                # self.f_ab_raw = Linear(in_channels, hidden_dim, device=device) # dtype=?
                # layer_init_kaiming_normal(self.f_ab_raw)
                self.f_ab_raw = AxialAttention(
                    dim=hidden_dim,               # embedding dimension
                    # dim_index = -1,         # where is the embedding dimension
                    dim_index = -1,         # where is the embedding dimension
                    dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                    heads = 1,             # number of heads for multi-head attention
                    num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
                    sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
                )
                # self.f_ab_raw = ViT(
                #     image_size = 600,
                #     patch_size = 1,
                #     num_classes = 51,
                #     dim = 1024,
                #     depth = 6,
                #     heads = 16,
                #     mlp_dim = 2048,
                #     dropout = 0.1,
                #     emb_dropout = 0.1,
                #     channels=768,
                # )
            if use_pairwise_l2 is True:
                # self.f_ab_obj_ctx = Linear(1, 1, device=device)
                # layer_init_kaiming_normal(self.f_ab_obj_ctx)
                self.f_ab_obj_ctx = AxialAttention(
                    dim=hidden_dim,               # embedding dimension
                    dim_index = -1,         # where is the embedding dimension
                    dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                    heads = 1,             # number of heads for multi-head attention
                    num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
                    sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
                )
                # self.f_ab_obj_ctx = ViT(
                #     image_size = 600,
                #     patch_size = 1,
                #     num_classes = 51,
                #     dim = 4096,
                #     depth = 6,
                #     heads = 16,
                #     mlp_dim = 2048,
                #     dropout = 0.1,
                #     emb_dropout = 0.1,
                #     channels=768,
                # )
            # self.f_ab_obj_ctx = ViT(image_size = 300, patch_size = 1, num_classes = 51, dim = 768, depth = 6, heads = 16, mlp_dim = 2048, dropout = 0.1, emb_dropout = 0.1).cuda()
            if use_pairwise_l3 is True:
                # self.f_ab_obj_ctx_subj_obj = Linear(1, 1, device=device)
                # layer_init_kaiming_normal(self.f_ab_obj_ctx_subj_obj)
                self.f_ab_obj_ctx_subj_obj = AxialAttention(
                    dim=hidden_dim,               # embedding dimension
                    dim_index = -1,         # where is the embedding dimension
                    dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
                    heads = 1,             # number of heads for multi-head attention
                    num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
                    sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
                )
                # self.f_ab_obj_ctx_subj_obj = ViT(
                #     image_size = 600,
                #     patch_size = 1,
                #     num_classes = 51,
                #     dim = 1024,
                #     depth = 6,
                #     heads = 16,
                #     mlp_dim = 2048,
                #     dropout = 0.1,
                #     emb_dropout = 0.1,
                #     channels=768,
                # )
        elif f_nn_type == 'mha':
            raise NotImplementedError(f'{f_nn_type} is not implementated')
        elif f_nn_type == 'lra':
            raise NotImplementedError(f'{f_nn_type} is not implementated')
        elif f_nn_type == 'gnn':
            raise NotImplementedError(f'{f_nn_type} is not implementated')
        elif f_nn_type == 'scl':
            raise NotImplementedError(f'{f_nn_type} is not implementated')
        else:
            raise NotImplementedError(f'{f_nn_type} is not implementated')
        # ROI feature fusion of many derived only from roi features
        self.fusion_method_roi = fusion_method_roi = cfg.MODEL.ROI_RELATION_HEAD.PAIRWISE.FUSION_METHOD_ROI
        # if fusion_method_roi == 'sum':
        #     self.after_fusion = Linear(hidden_dim, hidden_dim, device=device)
        #     self.bn_after_fusion = BatchNorm1d(hidden_dim, device=device)
        #     # ctx_pairwise_rois = obj_features_subject + obj_features_object + pairwise_ctx_raw + pairwise_ctx_obj_ctx + pairwise_ctx_obj_ctx_subj_obj
        #     # ctx_pairwise_rois = self.bn_after_fusion(self.act_after_fusion(self.after_fusion(ctx_pairwise_rois)))
        # elif fusion_method_roi == 'concat':
        #     self.after_fusion = Linear(hidden_dim, hidden_dim, device=device)
        #     self.bn_after_fusion = BatchNorm1d(hidden_dim, device=device)
        #     # ctx_pairwise_rois = torch_cat([obj_features_subject, obj_features_object, pairwise_ctx_raw, pairwise_ctx_obj_ctx, pairwise_ctx_obj_ctx_subj_obj])
        #     # ctx_pairwise_rois = self.bn_after_fusion(self.act_after_fusion(self.after_fusion(ctx_pairwise_rois)))
        # self.act_after_fusion = ReLU()

        # Final fusion from features derived from both roi features and union features
        self.rel_compress = Linear(pooling_dim, num_rel_cls, device=device)
        layer_init_kaiming_normal(self.rel_compress)
        self.ctx_compress = Linear(hidden_dim, num_rel_cls, device=device)
        layer_init_kaiming_normal(self.ctx_compress)

        self.fusion_method_final = fusion_method_final = cfg.MODEL.ROI_RELATION_HEAD.PAIRWISE.FUSION_METHOD_FINAL
        # if fusion_method_final == 'sum':
        #     self.fusion_final = Linear(pooling_dim, num_rel_cls, device=device)
        # elif fusion_method_final == 'concat':
        #     self.fusion_final = Linear(pooling_dim, num_rel_cls, device=device)
        # else:
        #     raise NotImplementedError(f'{fusion_method_final} is not implementated')
        # layer_init_kaiming_normal(self.fusion_final)

        print(memory_summary())


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, global_image_features=None):
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        # QUESTION: what is the purpose of the 1280 objects if we already have rel_pair_idxs?

        # rel_pair_idxs_global = torch_cat([rel_pair_idx + num_ ])
        rel_pair_idxs_global = [] # TODO: construct and fill instead?
        num_objs_culsum = 0
        # TODO: maybe use cumsum as an optimization?
        for rel_pair_idx, num_obj in zip(rel_pair_idxs, num_objs):
            rel_pair_idxs_global.append(rel_pair_idx + num_objs_culsum if num_objs_culsum > 0 else rel_pair_idx)
            num_objs_culsum += num_obj

        rel_pair_idxs_global = torch_cat(rel_pair_idxs_global)
        rel_pair_idxs_global_head = rel_pair_idxs_global[:, 0]
        rel_pair_idxs_global_tail = rel_pair_idxs_global[:, 1]
        del rel_pair_idxs_global
        # TODO: abs may not be necessary here because only the self multiplication is squared.
        # pairwise_raw = roi_features[:, None, :] * roi_features[None, :, :] # [1280, 1280, 4096] # or use roi_features instead
        if self.use_pairwise_l1 is True:
            pairwise_raw = roi_features[rel_pair_idxs_global_head, None, :] * roi_features[None, rel_pair_idxs_global_tail, :] # [3009, 3009, 4096] # or use roi_features instead
            pairwise_ctx_raw = self.bn_raw(self.act_pairwise_raw(self.f_ab_raw(pairwise_raw)))
            del pairwise_raw
            if self.use_pairwise_l2 is False and self.use_pairwise_l3 is False:
                del rel_pair_idxs_global_head, rel_pair_idxs_global_tail

        obj_ctx = self.bn_obj(self.act_obj(self.f_obj(roi_features))) # torch.Size([23, 768])
        del roi_features

        use_gt_label = self.training or self.use_gt_object_label
        obj_labels = torch_cat([proposal.get_field("labels").to(torch_int64, non_blocking=True) for proposal in proposals], dim=0) if use_gt_label else None

        if self.mode == 'predcls':
            del proposals
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
        else:
            obj_dists = self.out_obj(obj_ctx)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                del proposals
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                del proposals
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1

        obj_preds = obj_preds.split(num_objs, dim=0) # 16 * torch.Size([80])

        pair_preds = []
        for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
            pair_preds.append(obj_pred[pair_idx])

        pair_pred = torch_cat(pair_preds, dim=0) # torch.Size([3009, 2])
        del pair_preds

        obj_ctx_subj = obj_ctx[rel_pair_idxs_global_head] # [506, 768]
        obj_ctx_obj = obj_ctx[rel_pair_idxs_global_tail] # [506, 768]
        del obj_ctx

        if self.use_pairwise_l2 is True:
            # TODO: optimize the permute to reduce the number of permute operations.
            # pairwise_obj_ctx_old = obj_ctx[rel_pair_idxs_global_head, None, :] * obj_ctx[None, rel_pair_idxs_global_tail, :] # torch.Size([506, 506, 768]) # or use roi_features instead
            # TODO Need to normalize obj_ctx_subj obj_ctx_obj before multiplication because small numbers
            pairwise_obj_ctx = obj_ctx_subj.unsqueeze(1) * obj_ctx_obj.unsqueeze(0) # torch.Size([506, 506, 768]) # or use roi_features instead
            # breakpoint()
            # pairwise_ctx_obj_ctx = self.bn_pairwise_obj_ctx(self.act_pairwise_obj_ctx(self.f_ab_obj_ctx(pairwise_obj_ctx)))
            # x = self.f_ab_obj_ctx(pairwise_obj_ctx[None, :, :, :])
            # x = self.f_ab_obj_ctx.to_patch_embedding[0](pairwise_obj_ctx[None, :, :])
            # x = self.f_ab_obj_ctx.to_patch_embedding[1](x)
            # # x.size() = torch.Size([1, 256036, 4096])
            # b, n, _ = x.shape
            # from einops import repeat
            # cls_tokens = repeat(self.f_ab_obj_ctx.cls_token, '1 1 d -> b 1 d', b = b) # unnecessary
            # # cls_tokens.size(): torch.Size([1, 1, 4096])
            # # x.size() = torch.Size([1, 256036, 4096])
            # x = torch_cat((cls_tokens, x), dim=1) # torch.Size([1, 256037, 4096])
            # x += self.f_ab_obj_ctx.pos_embedding[:, :(n + 1)] # self.f_ab_obj_ctx.pos_embedding.size(): torch.Size([1, 90001, 4096]). Is pos_embedding correct?
            # breakpoint()
            # # This is why the image_size needs to be bigger. Otherwise pos_embedding will be smaller than actual
            # x = self.dropout(x)
            # x = self.transformer(x)
            # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
            # x = self.to_latent(x)
            # x = self.mlp_head(x)

            # x = self.f_ab_obj_ctx(pairwise_obj_ctx[None, :, :]) # torch.Size([1, 506, 506, 768])
            # breakpoint()
            # self.f_ab_obj_ctx(pairwise_obj_ctx)
            # pairwise_ctx_obj_ctx = self.bn_pairwise_obj_ctx(self.act_pairwise_obj_ctx(self.f_ab_obj_ctx(pairwise_obj_ctx[None, :, :])))
            # TODO Need to normalize obj_ctx_subj obj_ctx_obj before multiplication because small numbers
            # Cross product measures interactions in higher dimensions
            # TODO: use torch.linalg.cross() or is hadardmard ok?
            # Alternatively, can just do matrix multiplication with .T
            # Experiments: Pairwise Construction Variant 1. (This one) pairwise hadamard product
            # Experiments: Pairwise Construction Variant 2. Pairwise cross product
            # Experiments: Pairwise Construction Variant 3. Pairwise dot product
            # Experiments: Pairwise Construction Variant 4. Matrix Multiplication
            # NOTE: there may be opportunities for optimization in terms of combinatorics and permutations (duplicate across the diagonal?).
            # breakpoint()
            pairwise_ctx_obj_ctx = self.bn_pairwise_obj_ctx(self.act_pairwise_obj_ctx(self.f_ab_obj_ctx(pairwise_obj_ctx.unsqueeze(0))).permute(0, 3, 1, 2)).squeeze(0)
            # pairwise_ctx_obj_ctx = self.bn_pairwise_obj_ctx(self.act_pairwise_obj_ctx(self.f_ab_obj_ctx(pairwise_obj_ctx)).permute(0, 3, 1, 2)).squeeze(0)
            # torch.Size([768, 506, 506])
            # dim reduction
            # Experiments: Pairwise Summary Variant 1. (This one) sum across
            # Experiments: Pairwise Summary Variant 2. max but probably on the l2 norm of the 768 features. This is a problem because there are few unique values
            # Experiments: Pairwise Summary Variant 3. prod
            # breakpoint()
            # norms = vector_norm(pairwise_ctx_obj_ctx, dim=0, keepdim=False, dtype=torch_float32)
            # breakpoint()
            # idx = norms.argmax(dim=1, keepdim=False)
            # pairwise_ctx_obj_ctx = pairwise_ctx_obj_ctx.sum(dim=1, keepdim=False, dtype=torch_float32)
            # breakpoint()
            # pairwise_ctx_obj_ctx = pairwise_ctx_obj_ctx[idx]
            pairwise_ctx_obj_ctx = pairwise_ctx_obj_ctx.sum(dim=1) # [768, 506]
            # breakpoint()
            # TODO: CNN? It won't work because of non-region dependencies within the first two dimentions. Try GGNN and VIT instead.
            # I don't think I can use ggnn because ggnn is 2D. Let's do VIT then.
            # self.ggnn1()
            # self.vit1()
            # breakpoint()
            del pairwise_obj_ctx
            if self.use_pairwise_l3 is False:
                del rel_pair_idxs_global_head, rel_pair_idxs_global_tail
        obj_features_subject = self.bn_obj_subj(self.act_obj_subj(self.f_a(obj_ctx_subj)))
        # del obj_ctx_subj
        # breakpoint()

        obj_features_object = self.bn_obj_obj(self.act_obj_obj(self.f_b(obj_ctx_obj)))
        # del obj_ctx_obj
        # breakpoint()

        if self.use_pairwise_l3 is True:
            pairwise_obj_ctx_subj_obj = obj_features_subject[rel_pair_idxs_global_head, None, :] * obj_features_object[None, rel_pair_idxs_global_tail, :] # [1280, 1280, 4096] # or use roi_features instead
            # breakpoint()
            del rel_pair_idxs_global_head, rel_pair_idxs_global_tail
            pairwise_ctx_obj_ctx_subj_obj = self.bn_pairwise_ctx_subj_obj(self.act_pairwise_ctx_subj_obj(self.f_ab_obj_ctx_subj_obj(pairwise_obj_ctx_subj_obj)))
            # breakpoint()

            del pairwise_obj_ctx_subj_obj

        # Fusion
        # TODO: sum vs concat
        if self.fusion_method_roi == 'sum':
            # breakpoint()
            # ctx_pairwise_rois = obj_features_subject + obj_features_object + pairwise_ctx_raw + pairwise_ctx_obj_ctx + pairwise_ctx_obj_ctx_subj_obj
            # ctx_pairwise_rois = obj_features_subject + obj_features_object + pairwise_ctx_obj_ctx + pairwise_ctx_obj_ctx_subj_obj
            ctx_pairwise_rois = obj_features_subject + obj_features_object + pairwise_ctx_obj_ctx.transpose_(0, 1)
            # breakpoint()
            # breakpoint()
            # ctx_pairwise_rois = self.after_fusion(ctx_pairwise_rois) # [506, 768]
            # ctx_pairwise_rois = self.act_after_fusion(ctx_pairwise_rois)
            # ctx_pairwise_rois = self.bn_after_fusion(ctx_pairwise_rois)
            # ctx_pairwise_rois = self.bn_after_fusion(self.act_after_fusion(self.after_fusion(ctx_pairwise_rois)))
        elif self.fusion_method_roi == 'concat':
            ctx_pairwise_rois = torch_cat([obj_features_subject, obj_features_object, pairwise_ctx_raw, pairwise_ctx_obj_ctx, pairwise_ctx_obj_ctx_subj_obj]) # [506, 768]
            ctx_pairwise_rois = self.bn_after_fusion(self.act_after_fusion(self.after_fusion(ctx_pairwise_rois)))

        # breakpoint()

        # ctx_gate = self.post_cat(pairwise_ctx_obj_ctx) # torch.Size([3009, 4096]) # TODO: Use F_sigmoid?
        # breakpoint()
        ctx_gate = self.post_cat(torch_cat((obj_ctx_subj, obj_ctx_obj), dim=-1)) # torch.Size([3009, 4096]) # TODO: Use F_sigmoid?
        del obj_ctx_subj, obj_ctx_obj
        # breakpoint()
        visual_rep = union_features * ctx_gate
        del union_features, ctx_gate
        # breakpoint()

        if self.fusion_method_final == 'sum':
            # breakpoint()
            rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(ctx_pairwise_rois)
        elif self.fusion_method_final == 'concat':
            rel_dists = torch_cat(self.rel_compress(visual_rep), self.ctx_compress(ctx_pairwise_rois))
        else:
            raise NotImplementedError(f'{self.fusion_method_final} is not implemented')
        if self.use_bias: # True
            freq_dists_bias = self.freq_bias.index_with_labels(pair_pred)
            del pair_pred
            rel_dists += self.dropout(freq_dists_bias) # torch.Size([3009, 51])

        # TODO: instead use pairwise_ctx_obj_ctx because we already have the rel_pair_idxs
        return obj_dists.split(num_objs), rel_dists.split(num_rels, dim=0), {}


    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)

            out_dists_sampled = F_softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new_full(num_objs[i], 0, dtype=torch_int64, device=self.device)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np_unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label)
        return torch_cat(obj_preds, dim=0)


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
