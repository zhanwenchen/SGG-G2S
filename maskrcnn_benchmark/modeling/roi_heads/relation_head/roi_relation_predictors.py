# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# import numpy as np
# from time import time_ns as time_time_ns
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
)
from torch.utils.benchmark import Timer, Compare
from torch.jit import script as torch_jit_script
from torch.nn import Module, Sequential, Linear, ReLU
from torch.nn.functional import dropout as F_dropout, binary_cross_entropy_with_logits as F_binary_cross_entropy_with_logits, relu as F_relu, softmax as F_softmax, cross_entropy as F_cross_entropy
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.layers.gcn._utils import adj_normalize
from .model_motifs import FrequencyBias, to_onehot
from .model_transformer import TransformerContext, TransformerEncoder
from .utils_relation import MLP
# from util_misc import print_para


# from util_debug import gpu_profile

# from sys import settrace, _getframe

# settrace(gpu_profile)
# from torch import is_tensor as torch_is_tensor
# from gc import get_objects
# # prints currently alive Tensors and Variables
# def debug_memory():
#     print('debug_memory:')
#     for obj in get_objects():
#         try:
#             if torch_is_tensor(obj) or (hasattr(obj, 'data') and torch_is_tensor(obj)):
#                 print(type(obj), obj.size())
#         except:
#             pass
#     print()

from torch import is_tensor as torch_is_tensor
from gc import get_objects
def debug_memory():
    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             print(type(obj), obj.size())
    #     except:
    #         pass
    for obj in get_objects():
        if torch_is_tensor(obj):
            print(obj.numel() if len(obj.size()) > 0 else 0, type(obj), obj.size())


@registry.ROI_RELATION_PREDICTOR.register("TransformerTransferGSCPredictor")
class TransformerTransferGSCPredictor(Module):
    def __init__(self, config, in_channels):
        super(TransformerTransferGSCPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters`
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE

        assert in_channels is not None
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.with_knowdist = False
        self.devices = config.MODEL.DEVICE
        device = torch_device(self.devices)
        self.with_transfer = config.MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER
        self.with_cleanclf = config.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER
        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        print(f'rel_classes={rel_classes}')
        self.val_alpha = config.MODEL.ROI_RELATION_HEAD.VAL_ALPHA
        mode = None
        use_gt_box = config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX
        use_gt_object_label = config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        if use_gt_box is True and use_gt_object_label is True:
            mode = 'predcls'
        if use_gt_box is True and use_gt_object_label is False:
            mode = 'sgcls'
        if use_gt_box is False and use_gt_object_label is False:
            mode = 'sgdet'
        if mode is None:
            raise ValueError(f'mode is None given use_gt_box={use_gt_box} and use_gt_object_label={use_gt_object_label}')
        self.mode = mode
        # # module construct
        # self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        # self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        # self.v_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM
        # self.k_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        # self.inner_dim = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        # self.num_head = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        # self.edge_layer = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        #
        # # New
        # self.context_gsc = TransformerEncoder(self.edge_layer, self.num_head, self.k_dim, self.v_dim, self.pooling_dim, self.inner_dim, self.dropout_rate)
        # self.post_emb_gsc = Linear(self.pooling_dim, self.pooling_dim)
        # # layer_init(, 10.0 * (1.0 / self.pooling_dim) ** 0.5, normal=True)
        # layer_init_kaiming_normal(self.post_emb_gsc)
        # # module construct
        # self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)
        #
        # # post decoding
        #
        # self.post_emb = Linear(self.hidden_dim, self.hidden_dim * 2)
        # layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        # self.post_cat = Linear(self.hidden_dim * 2, self.pooling_dim)
        # layer_init(self.post_cat, xavier=True)
        #
        # if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
        #     self.union_single_not_match = True
        #     self.up_dim = Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
        #     layer_init(self.up_dim, xavier=True)
        # else:
        #     self.union_single_not_match = False
        #
        # # initialize layer parameters
        # self.rel_compress = Linear(self.pooling_dim, self.num_rel_cls)
        # self.ctx_compress = Linear(self.hidden_dim * 2, self.num_rel_cls)
        # self.gsc_compress = Linear(self.pooling_dim, self.num_rel_cls)
        # layer_init(self.rel_compress, xavier=True)
        # layer_init(self.ctx_compress, xavier=True)
        # layer_init(self.gsc_compress, xavier=True)
        # if self.use_bias:
        #     # convey statistics into FrequencyBias to avoid loading again
        #     self.freq_bias = FrequencyBias(config, statistics)
        #
        # # the transfer classifier
        # if self.with_cleanclf:
        #     self.rel_compress_clean = Linear(self.pooling_dim, self.num_rel_cls)
        #     self.ctx_compress_clean = Linear(self.hidden_dim * 2, self.num_rel_cls)
        #     self.gsc_compress_clean = Linear(self.pooling_dim, self.num_rel_cls)
        #     layer_init(self.rel_compress_clean, xavier=True)
        #     layer_init(self.ctx_compress_clean, xavier=True)
        #     layer_init(self.gsc_compress_clean, xavier=True)
        #     # self.gcns_rel_clean = GCN(self.pooling_dim, self.pooling_dim, self.dropout_rate)
        #     # self.gcns_ctx_clean = GCN(self.hidden_dim * 2, self.hidden_dim * 2, self.dropout_rate)
        #     self.freq_bias_clean = FrequencyBias(config, statistics)
        # if self.with_transfer:
        #     #pred_adj_np = np.load('./misc/conf_mat_adj_mat.npy')
        #     print("Using Confusion Matrix Transfer!")
        #     pred_adj_np = np.load('./misc/conf_mat_freq_train.npy')
        #     # pred_adj_np = 1.0 - pred_adj_np
        #     pred_adj_np[0, :] = 0.0
        #     pred_adj_np[:, 0] = 0.0
        #     pred_adj_np[0, 0] = 1.0
        #     # adj_i_j means the baseline outputs category j, but the ground truth is i.
        #     pred_adj_np = pred_adj_np / (pred_adj_np.sum(-1)[:, None] + 1e-8)
        #     pred_adj_np = adj_normalize(pred_adj_np)
        #     self.pred_adj_nor = torch_from_numpy(pred_adj_np).float().to(self.devices)
        #     # self.pred_adj_layer_clean = Linear(self.num_rel_cls, self.num_rel_cls, bias=False)
        #     # #layer_init(self.pred_adj_layer_clean, xavier=True)
        #     # with torch_no_grad():
        #     #     self.pred_adj_layer_clean.weight.copy_(torch.eye(self.num_rel_cls,dtype=torch.float), non_blocking=True)
        #         #self.pred_adj_layer_clean.weight.copy_(self.pred_adj_nor, non_blocking=True)
        if config.MODEL.ROI_RELATION_HEAD.GBNET.USE_EMBEDDING is True:
            with open(config.MODEL.ROI_RELATION_HEAD.GBNET.EMB_PATH, 'rb') as fin:
                self.emb_ent, self.emb_pred = pickle_load(fin)
        # else:
        #     self.emb_ent = torch_eye(num_ents, dtype=torch_float32)
        #     self.emb_pred = torch_eye(num_preds, dtype=torch_float32)

        # self.num_ont_ent = self.emb_ent.size(0)
        # assert self.num_ont_ent == num_obj_cls
        # self.num_ont_pred = self.emb_pred.size(0)
        # assert self.num_ont_pred == num_rel_cls

        if config.MODEL.ROI_RELATION_HEAD.GBNET.USE_KNOWLEDGE is True:
            with open(config.MODEL.ROI_RELATION_HEAD.GBNET.KB_PATH, 'rb') as fin:
                edge_dict = pickle_load(fin)

            self.adjmtx_ent2ent = edge_dict['edges_ent2ent']
            self.adjmtx_ent2pred = edge_dict['edges_ent2pred']
            self.adjmtx_pred2ent = edge_dict['edges_pred2ent']
            self.adjmtx_pred2pred = edge_dict['edges_pred2pred']

        self.time_step_num = config.MODEL.ROI_RELATION_HEAD.GBNET.TIME_STEP_NUM
        # else:
        #     self.adjmtx_ent2ent = np.zeros((1, num_ents, num_ents), dtype=np.float32)
        #     self.adjmtx_ent2pred = np.zeros((1, num_ents, num_preds), dtype=np.float32)
        #     self.adjmtx_pred2ent = np.zeros((1, num_preds, num_ents), dtype=np.float32)
        #     self.adjmtx_pred2pred = np.zeros((1, num_preds, num_preds), dtype=np.float32)

        self.num_edge_types_ent2ent = self.adjmtx_ent2ent.shape[0]
        self.num_edge_types_ent2pred = self.adjmtx_ent2pred.shape[0]
        self.num_edge_types_pred2ent = self.adjmtx_pred2ent.shape[0]
        self.num_edge_types_pred2pred = self.adjmtx_pred2pred.shape[0]

        hidden_dim = config.MODEL.ROI_RELATION_HEAD.GBNET.HIDDEN_DIM
        self.fc_init_ont_ent = Linear(self.emb_ent.shape[1], hidden_dim) # 300, 1024
        self.fc_init_ont_pred = Linear(self.emb_pred.shape[1], hidden_dim)

        self.fc_mp_send_ont_ent = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True, device=device)
        self.fc_mp_send_ont_pred = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True, device=device)
        self.fc_mp_send_img_ent = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True, device=device)
        self.fc_mp_send_img_pred = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True, device=device)

        self.fc_mp_receive_ont_ent = MLP([(self.num_edge_types_ent2ent + self.num_edge_types_pred2ent + 1) * hidden_dim // 4,
                                          (self.num_edge_types_ent2ent + self.num_edge_types_pred2ent + 1) * hidden_dim // 4,
                                          hidden_dim], act_fn='ReLU', last_act=True, device=device)
        self.fc_mp_receive_ont_pred = MLP([(self.num_edge_types_ent2pred + self.num_edge_types_pred2pred + 1) * hidden_dim // 4,
                                           (self.num_edge_types_ent2pred + self.num_edge_types_pred2pred + 1) * hidden_dim // 4,
                                           hidden_dim], act_fn='ReLU', last_act=True, device=device)
        self.fc_mp_receive_img_ent = MLP([3 * hidden_dim // 4, 3 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True, device=device)
        self.fc_mp_receive_img_pred = MLP([3 * hidden_dim // 4, 3 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True, device=device)

        self.fc_eq3_w_ont_ent = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq3_u_ont_ent = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq4_w_ont_ent = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq4_u_ont_ent = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq5_w_ont_ent = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq5_u_ont_ent = Linear(hidden_dim, hidden_dim, device=device)

        self.fc_eq3_w_ont_pred = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq3_u_ont_pred = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq4_w_ont_pred = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq4_u_ont_pred = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq5_w_ont_pred = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq5_u_ont_pred = Linear(hidden_dim, hidden_dim, device=device)

        self.fc_eq3_w_img_ent = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq3_u_img_ent = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq4_w_img_ent = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq4_u_img_ent = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq5_w_img_ent = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq5_u_img_ent = Linear(hidden_dim, hidden_dim, device=device)

        self.fc_eq3_w_img_pred = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq3_u_img_pred = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq4_w_img_pred = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq4_u_img_pred = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq5_w_img_pred = Linear(hidden_dim, hidden_dim, device=device)
        self.fc_eq5_u_img_pred = Linear(hidden_dim, hidden_dim, device=device)

        self.fc_output_proj_img_pred = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False, device=device)
        self.fc_output_proj_ont_pred = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False, device=device)

        # TODO: isn't refine_obj_cls determined by mode? Let's just do config now
        self.refine_obj_cls = config.MODEL.ROI_RELATION_HEAD.GBNET.REFINE_OBJ_CLS
        if self.refine_obj_cls:
            self.fc_output_proj_img_ent = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False, device=device)
            self.fc_output_proj_ont_ent = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False, device=device)

        self.obj_dim = 4096
        self.rel_dim = 4096
        self.obj_proj = Linear(self.obj_dim, hidden_dim)
        self.rel_proj = Linear(self.rel_dim, hidden_dim)


        # else:
            # self.rel_class_weights = np_ones((self.num_rels,))

        # self.debug_info = {}

        # self.use_ontological_adjustment = config.MODEL.USE_ONTOLOGICAL_ADJUSTMENT
        # self.normalize_eoa = config.MODEL.NORMALIZE_EOA
        # self.shift_eoa = config.MODEL.SHIFT_EOA
        # self.fold_eoa = config.MODEL.FOLD_EOA
        # self.merge_eoa_sa = config.MODEL.MERGE_EOA_SA

        # if self.use_ontological_adjustment is True:
        #     # 4x51x51 => 51x51
        #     print('my_ggnn_10: using use_ontological_adjustment')
        #     # ontological_preds = torch_tensor(self.adjmtx_pred2pred, dtype=torch_float32, device=CUDA_DEVICE).sum(axis=0)
        #     ontological_preds = self.adjmtx_pred2pred[:3, :, :].sum(axis=0)
        #     ontological_preds[0, :] = 0.0
        #     ontological_preds[:, 0] = 0.0
        #     ontological_preds[0, 0] = 1.0
        #     if self.fold_eoa is True:
        #         diag_indices = np.diag_indices(ontological_preds.shape[0])
        #         folded = ontological_preds + ontological_preds.T
        #         folded[diag_indices] = ontological_preds[diag_indices]
        #     if self.shift_eoa is True:
        #         ontological_preds += 1.0
        #         print(f'EOA-N: Used shift_eoa')
        #     else:
        #         print(f'EOA-N: Not using shift_eoa. self.eoa_n={self.normalize_eoa}')
        #     ontological_preds = ontological_preds / (ontological_preds.sum(-1)[:, None] + 1e-8)
        #     if self.normalize_eoa is True:
        #         ontological_preds = adj_normalize(ontological_preds)
        #         print(f'EOA-N: Used adj_normalize')
        #     else:
        #         print(f'EOA-N: Not using adj_normalize. self.eoa_n={self.normalize_eoa}')
        #     self.ontological_preds = torch_tensor(ontological_preds, dtype=torch_float32, device=CUDA_DEVICE)
        # else:
        #     print(f'my_ggnn_10: not using use_ontological_adjustment. self.use_ontological_adjustment={self.use_ontological_adjustment}')

        # if self.with_clean_classifier:
        #     self.fc_output_proj_img_pred_clean = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
        #     self.fc_output_proj_ont_pred_clean = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
        #
        #     if self.refine_obj_cls:
        #         self.fc_output_proj_img_ent_clean = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
        #         self.fc_output_proj_ont_ent_clean = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)

        # if self.sa is True:
        #     print("!!!!!!!!!With Confusion Matrix Channel!!!!!")
        #     pred_adj_np = np.load(config.MODEL.CONF_MAT_FREQ_TRAIN)
        #     # pred_adj_np = 1.0 - pred_adj_np
        #     pred_adj_np[0, :] = 0.0
        #     pred_adj_np[:, 0] = 0.0
        #     pred_adj_np[0, 0] = 1.0
        #     # adj_i_j means the baseline outputs category j, but the ground truth is i.
        #     pred_adj_np = pred_adj_np / (pred_adj_np.sum(-1)[:, None] + 1e-8)
        #     pred_adj_np = adj_normalize(pred_adj_np)
        #     print(f'SA: Used adj_normalize')
        #     self.pred_adj_nor = torch_as_tensor(pred_adj_np, dtype=torch_float32, device=CUDA_DEVICE)


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
        device = union_features.device
        # breakpoint()
        ent_cls_logits_all = []
        pred_cls_logits_all = []
        # idx_start_ent = 0
        # idx_start_pred = 0

        roi_features = self.obj_proj(roi_features) # torch.Size([85, 4096]) => torch.Size([85, 1024])
        union_features = self.rel_proj(union_features) # torch.Size([1138, 4096]) => torch.Size([1138, 1024])

        if proposals is None:
            print(f'proposals is None')
            breakpoint()

        if rel_pair_idxs is None:
            print(f'rel_pair_idxs is None')
            breakpoint()
        num_objs = [len(proposal) for proposal in proposals]
        num_rels = [len(rel_pair_idx) for rel_pair_idx in rel_pair_idxs]
        roi_features_images = roi_features.split(num_objs)
        union_features_images = union_features.split(num_rels)

        for img_idx, (proposal, rel_pair_idx, roi_features_image, union_features_image) in enumerate(zip(proposals, rel_pair_idxs, roi_features_images, union_features_images)):
            '''
            len(proposal) = 12. All proposals sum to 94 which is the nrows of roi_features
            rel_pair_idx.shape = torch.Size([132, 2])
            len(rel_label) = 132

            '''
            ent_cls_logits, pred_cls_logits = self.per_image(proposal, rel_pair_idx, roi_features_image, union_features_image, device=device, debug=True)

            ent_cls_logits_all.append(ent_cls_logits)
            pred_cls_logits_all.append(pred_cls_logits)
        return ent_cls_logits_all, pred_cls_logits_all, {}
        #     obj_labels = proposal.get_field("labels")
        #
        #     if self.mode == 'predcls':
        #         obj_preds = obj_labels
        #         obj_dists = to_onehot(obj_preds, self.num_obj_cls)
        #     # breakpoint()
        #     # if self.mode == 'predcls':
        #     #     obj_preds = to_onehot(obj_preds, self.num_obj_cls)
        #
        #     num_objs = len(proposal) #0: 15
        #     num_rels = len(rel_pair_idx) #0: 210
        #
        #     nodes_ont_ent = self.fc_init_ont_ent(torch_as_tensor(self.emb_ent, dtype=torch_float32, device=device))
        #     nodes_ont_pred = self.fc_init_ont_pred(torch_as_tensor(self.emb_pred, dtype=torch_float32, device=device))
        #     nodes_img_ent = roi_features[idx_start_ent:idx_start_ent+num_objs]
        #     idx_start_ent += num_objs
        #     nodes_img_pred = union_features[idx_start_pred:idx_start_pred+num_rels]
        #     idx_start_pred += num_rels
        #
        #     # TODO: use placeholders instead of generating new tensors which
        #     # require a recreation of the computation graph. This may be true
        #     # even if requires_grad = False.
        #     edges_ont_ent2ent = torch_as_tensor(self.adjmtx_ent2ent, dtype=torch_float32, device=device)
        #     edges_ont_ent2pred = torch_as_tensor(self.adjmtx_ent2pred, dtype=torch_float32, device=device)
        #     edges_ont_pred2ent = torch_as_tensor(self.adjmtx_pred2ent, dtype=torch_float32, device=device)
        #     edges_ont_pred2pred = torch_as_tensor(self.adjmtx_pred2pred, dtype=torch_float32, device=device)
        #
        #     edges_img_pred2subj = torch_zeros((num_rels, num_objs), dtype=torch_float32, device=device)
        #     edges_img_pred2subj[:, rel_pair_idx[:, 0]] = 1
        #     edges_img_pred2obj = torch_zeros((num_rels, num_objs), dtype=torch_float32, device=device)
        #     edges_img_pred2obj[:, rel_pair_idx[:, 1]] = 1
        #     edges_img_subj2pred = edges_img_pred2subj.t() # TODO: need to specify axes when vectorized
        #     edges_img_obj2pred = edges_img_pred2obj.t()
        #
        #     edges_img2ont_ent = obj_dists.clone().detach()
        #     edges_ont2img_ent = edges_img2ont_ent.t()
        #
        #     num_rel_cls = self.num_rel_cls
        #     edges_img2ont_pred = torch_zeros((num_rels, num_rel_cls), dtype=torch_float32, device=device, requires_grad=False) # torch.Size([506, 51])
        #     edges_ont2img_pred = edges_img2ont_pred.t()
        #
        #     ent_cls_logits = None
        #
        #     num_edge_types_ent2ent = self.num_edge_types_ent2ent
        #     num_edge_types_pred2ent = self.num_edge_types_pred2ent
        #     num_edge_types_ent2pred = self.num_edge_types_ent2pred
        #     num_edge_types_pred2pred = self.num_edge_types_pred2pred
        #
        #     num_threads = torch_get_num_threads()
        #     print(f'Benchmarking on {num_threads} threads')
        #
        #     results = []
        #
        #     # with_clean_classifier = self.with_clean_classifier
        #
        #     for t in range(self.time_step_num):
        #         # breakpoint()
        #         message_send_ont_ent = self.fc_mp_send_ont_ent(nodes_ont_ent) # torch.Size([177, 1024]) => torch.Size([177, 256])
        #         # breakpoint()
        #         message_send_ont_pred = self.fc_mp_send_ont_pred(nodes_ont_pred) # torch.Size([51, 1024]) => torch.Size([51, 256])
        #         # breakpoint()
        #         message_send_img_ent = self.fc_mp_send_img_ent(nodes_img_ent) # torch.Size([23, 1024]) => torch.Size([23, 256])
        #         # breakpoint()
        #         message_send_img_pred = self.fc_mp_send_img_pred(nodes_img_pred) # torch.Size([506, 1024]) => torch.Size([506, 256])
        #         # breakpoint()
        #         message_received_ont_ent = torch_cat(
        #             [torch_mm(edges_ont_ent2ent[i].t(), message_send_ont_ent) for i in range(num_edge_types_ent2ent)] + # 177, 177 => 177, 256 x 9
        #             [torch_mm(edges_ont_pred2ent[i].t(), message_send_ont_pred) for i in range(num_edge_types_pred2ent)] + # 51 177 => 51, 256 * 3
        #             [torch_mm(edges_img2ont_ent.t(), message_send_img_ent),]
        #         , 1) #  torch.Size([177, 3328])
        #         # NOTE: there's some vectorization opportunity right here.
        #         # breakpoint()
        #         message_received_ont_ent = self.fc_mp_receive_ont_ent(message_received_ont_ent) # message_received_ont_ent: torch.Size([177, 3328]) => torch.Size([177, 1024])
        #
        #         # edges_ont_ent2pred: torch.Size([3, 151, 51])
        #         # message_send_ont_ent: torch.Size([151, 256])
        #         # og = torch_cat([torch_mm(edges_ont_ent2pred[i].t(), message_send_ont_ent) for i in range(num_edge_types_ent2pred)], 1)
        #         # # og: 51, 768
        #         # breakpoint()
        #         # lol = torch_einsum('abc,bd->cad', edges_ont_ent2pred, message_send_ont_ent).view(num_rel_cls, -1) # 3, 51, 256
        #
        #         # import torch
        #         # # b=3, n=151, m=51; b=1, m=51, p=256; x
        #         # # b=51, n=3, m=151, b=51, m=151, p=1; v
        #         # lol = edges_ont_ent2pred.view(3, 151, 51)
        #         # message_send_ont_pred.unsqueeze(1)
        #         # lol = torch.movedim(edges_ont_ent2pred, 2, 0)
        #         # TODO:
        #         # [a=3, b=151, c=51], [151, 256] => [3, 51, 256]
        #         # a = torch.rand(3,151,51)
        #         # b = torch.rand(151,256)
        #         # c = torch.cat([torch.mm(a[zi].t(), b) for i in range(3)], 1)
        #         # c1 = torch.mm(a[0].t(), b)
        #         #
        #         # lol = torch.einsum('abc,bd->cad', edges_ont_ent2pred, message_send_ont_ent)
        #         # # c2 = torch.einsum('abc,bd->acd', a, b)
        #         # c2 = torch.einsum('abc,bd->cad', a, b)
        #         # c2v = c2.view(51, 768)
        #         # # c1n = c2[0, :, :]
        #         # # cn = c2.permute(1, 0, 2).reshape(51, 768)
        #         # # torch.equal(c1, c1n)
        #         # # torch.equal(c, cn)
        #         # lol: torch.Size([3, 51, 256])
        #
        #         # [a=3, b=151, c=51], [151, 256] => [3, 51, 256]
        #         # torch.einsum("abc,bd->acd")
        #         # lol = torch.movedim(edges_ont_ent2pred, 2, 0) # not contiguous
        #         # torch.bmm(edges_ont_ent2pred.moveaxis(), )
        #         # torch.bmm(message_send_ont_ent.unsqueeze(-1))
        #         # ijk = [3, 151, 51]; [51, 256]
        #         # i = 3, j = 151, k = 51,
        #         # jk
        #         # lol = torch.einsum('ijk, kl -> ikl', edges_ont_ent2pred, message_send_ont_ent)
        #         # lol = torch.einsum('ijk, nkl -> kli', edges_ont_ent2pred.view(51, ), message_send_ont_ent.unsqueeze(0))
        #         # edges_ont_ent2pred: 3, 151, 51
        #         # edges_ont_ent2pred[0] # 151, 51
        #         # message_send_ont_ent: 151, 126
        #         # torch_mm(edges_ont_ent2pred[i].t(), message_send_ont_ent) # torch.Size([51, 256])
        #         # new = torch_mm(edges_ont_ent2pred, message_send_ont_ent)
        #         message_received_ont_pred = torch_cat(
        #             [torch_mm(edges_ont_ent2pred[i].t(), message_send_ont_ent) for i in range(num_edge_types_ent2pred)] +
        #             [torch_mm(edges_ont_pred2pred[i].t(), message_send_ont_pred) for i in range(num_edge_types_pred2pred)] +
        #             [torch_mm(edges_img2ont_pred.t(), message_send_img_pred),] # edges_ont2img_pred: torch.Size([51, 506]) @ torch.Size([506, 256])
        #         , 1) # torch.Size([51, 2048]) # Key is here. Why is this set?
        #         message_received_ont_pred = self.fc_mp_receive_ont_pred(message_received_ont_pred) # torch.Size([51, 2048]) => torch.Size([51, 1024])
        #
        #         # breakpoint()
        #         message_received_img_ent = torch_cat([
        #             torch_mm(edges_img_pred2subj.t(), message_send_img_pred),
        #             torch_mm(edges_img_pred2obj.t(), message_send_img_pred),
        #             torch_mm(edges_ont2img_ent.t(), message_send_ont_ent),
        #         ], 1) # torch.Size([23, 768])
        #         message_received_img_ent = self.fc_mp_receive_img_ent(message_received_img_ent) # torch.Size([23, 768]) => torch.Size([23, 1024])
        #         # import pdb; pdb.set_trace()
        #
        #         del message_send_ont_ent, message_send_img_pred
        #
        #         # breakpoint()
        #         message_received_img_pred = torch_cat([
        #             torch_mm(edges_img_subj2pred.t(), message_send_img_ent),
        #             torch_mm(edges_img_obj2pred.t(), message_send_img_ent),
        #             torch_mm(edges_ont2img_pred.t(), message_send_ont_pred),
        #         ], 1) # torch.Size([with_clean_classifier506, 768])
        #         message_received_img_pred = self.fc_mp_receive_img_pred(message_received_img_pred) # torch.Size([506, 768])=>torch.Size([506, 1024])
        #
        #         del message_send_ont_pred, message_send_img_ent
        #
        #         r_ont_ent = addsigmoid(self.fc_eq4_w_ont_ent(message_received_ont_ent), self.fc_eq4_u_ont_ent(nodes_ont_ent))
        #         nodes_ont_ent = formula(nodes_ont_ent,
        #             self.fc_eq3_w_ont_ent(message_received_ont_ent),
        #             self.fc_eq3_u_ont_ent(nodes_ont_ent),
        #             self.fc_eq5_w_ont_ent(message_received_ont_ent),
        #             self.fc_eq5_u_ont_ent(r_ont_ent * nodes_ont_ent),
        #         )
        #         del r_ont_ent
        #
        #         r_ont_pred = addsigmoid(self.fc_eq4_w_ont_pred(message_received_ont_pred), self.fc_eq4_u_ont_pred(nodes_ont_pred))
        #         nodes_ont_pred = formula(nodes_ont_pred,
        #             self.fc_eq3_w_ont_pred(message_received_ont_pred),
        #             self.fc_eq3_u_ont_pred(nodes_ont_pred),
        #             self.fc_eq5_w_ont_pred(message_received_ont_pred),
        #             self.fc_eq5_u_ont_pred(r_ont_pred * nodes_ont_pred),
        #         )
        #         del r_ont_pred
        #
        #         r_img_ent = addsigmoid(self.fc_eq4_w_img_ent(message_received_img_ent), self.fc_eq4_u_img_ent(nodes_img_ent))
        #         nodes_img_ent = formula(nodes_img_ent,
        #             self.fc_eq3_w_img_ent(message_received_img_ent),
        #             self.fc_eq3_u_img_ent(nodes_img_ent),
        #             self.fc_eq5_w_img_ent(message_received_img_ent),
        #             self.fc_eq5_u_img_ent(r_img_ent * nodes_img_ent),
        #         )
        #         del r_img_ent
        #
        #         # nodes_img_pred_old = nodes_img_pred.clone().detach()
        #         # message_received_img_pred_old = message_received_img_pred.clone().detach()
        #         # start_old = time_time_ns()
        #         # debug_memory()
        #         size = message_received_img_pred.size()
        #         timer_old = Timer(
        #             label='gating formula',
        #             sub_label=f'{size}_{t}',
        #             description='old',
        #             stmt='lol = self.get_new_nodes_img_pred_old(message_received_img_pred, nodes_img_pred)',
        #             setup='',
        #             globals={
        #                 'self': self,
        #                 'message_received_img_pred': message_received_img_pred,
        #                 'nodes_img_pred': nodes_img_pred,
        #             }
        #         ).blocked_autorange(min_run_time=1)
        #         results.append(timer_old)
        #         # print('OLD: 4x get_new_nodes_img_pred_old(message_received_img_pred, nodes_img_pred):')
        #         # print(timer_old.timeit(100))
        #         # print()
        #         # print(gpu_profile(frame=_getframe(), event='line', arg=None))
        #         print('timer_old:')
        #         print(print_para(self))
        #         # debug_memory()
        #
        #         timer_2x = Timer(
        #             label='gating formula',
        #             sub_label=f'{size}_{t}',
        #             description='2x',
        #             stmt='lol = self.get_new_nodes_img_pred_2x(message_received_img_pred, nodes_img_pred)',
        #             setup='',
        #             globals={
        #                 'self': self,
        #                 'message_received_img_pred': message_received_img_pred,
        #                 'nodes_img_pred': nodes_img_pred,
        #             }
        #         ).blocked_autorange(min_run_time=1)
        #         results.append(timer_2x)
        #         print('timer_2x:')
        #         print(print_para(self))
        #         # debug_memory()
        #         # print('NEW 2x: get_new_nodes_img_pred_2x(message_received_img_pred, nodes_img_pred):')
        #         # print(timer_2x.timeit(100))
        #         # print()
        #         # print(gpu_profile(frame=_getframe(), event='line', arg=None))
        #
        #         timer_2x_no_addsig = Timer(
        #             label='gating formula',
        #             sub_label=f'{size}_{t}',
        #             description='2x_no_addsig',
        #             stmt='lol = self.get_new_nodes_img_pred_2x_no_addsigmoid(message_received_img_pred, nodes_img_pred)',
        #             setup='',
        #             globals={
        #                 'self': self,
        #                 'message_received_img_pred': message_received_img_pred,
        #                 'nodes_img_pred': nodes_img_pred,
        #             }
        #         ).blocked_autorange(min_run_time=1)
        #         results.append(timer_2x_no_addsig)
        #         print('timer_2x_no_addsig:')
        #         print(print_para(self))
        #         # debug_memory()
        #         # print(gpu_profile(frame=_getframe(), event='line', arg=None))
        #
        #         timer_2x_no_addsig_oneline = Timer(
        #             label='gating formula',
        #             sub_label=f'{size}_{t}',
        #             description='2x_no_addsig_oneline',
        #             stmt='lol = self.get_new_nodes_img_pred_2x_no_addsigmoid_oneline(message_received_img_pred, nodes_img_pred)',
        #             setup='',
        #             globals={
        #                 'self': self,
        #                 'message_received_img_pred': message_received_img_pred,
        #                 'nodes_img_pred': nodes_img_pred,
        #             }
        #         ).blocked_autorange(min_run_time=1)
        #         results.append(timer_2x_no_addsig_oneline)
        #         print('timer_2x_no_addsig_oneline:')
        #         print(print_para(self))
        #         # debug_memory()
        #         # print(gpu_profile(frame=_getframe(), event='line', arg=None))
        #
        #         timer_2x_no_addsig_oneline_out_cls = Timer(
        #             label='gating formula',
        #             sub_label=f'{size}_{t}',
        #             description='2x_no_addsig_oneline_out_cls',
        #             stmt='lol = self.get_new_nodes_img_pred_2x_no_addsigmoid_oneline_out_cls(message_received_img_pred, nodes_img_pred)',
        #             setup='',
        #             globals={
        #                 'self': self,
        #                 'message_received_img_pred': message_received_img_pred,
        #                 'nodes_img_pred': nodes_img_pred,
        #             }
        #         ).blocked_autorange(min_run_time=1)
        #         results.append(timer_2x_no_addsig_oneline_out_cls)
        #         print('timer_2x_no_addsig_oneline_out_cls:')
        #         print(print_para(self))
        #         # debug_memory()
        #         # print(gpu_profile(frame=_getframe(), event='line', arg=None))
        #
        #         # timer_2x_no_addsig_oneline_out_fn = Timer(
        #         #     label='gating formula',
        #         #     sub_label=f'{size}_{t}',
        #         #     description='2x_no_addsig_oneline_out_fn',
        #         #     stmt='self.get_new_nodes_img_pred_2x_no_addsigmoid_oneline_out_fn(message_received_img_pred, nodes_img_pred)',
        #         #     setup='',
        #         #     globals={
        #         #         'self': self,
        #         #         'message_received_img_pred': message_received_img_pred,
        #         #         'nodes_img_pred': nodes_img_pred,
        #         #     }
        #         # ).blocked_autorange(min_run_time=1)
        #         # results.append(ti Timer(
        #         #     label='gating formula',
        #         #     sub_label=f'{size}_{t}',
        #         #     description='2x_no_addsig_oneline_out_fn',
        #         #     stmt='self.get_new_nodes_img_pred_2x_no_addsigmoid_oneline_out_fn(message_received_img_pred, nodes_img_pred)',
        #         #     setup='',
        #         #     globals={
        #         #         'self': self,
        #         #         'message_received_img_pred': message_received_img_pred,
        #         #         'nodes_img_pred': nodes_img_pred,
        #         #     }
        #         # ).blocked_autorange(min_run_time=1)
        #         # results.append(timemer_2x_no_addsig_oneline_out_fn)
        #         # print('NEW 2x_no_addsig: get_new_nodes_img_pred_2x_no_addsigmoid(message_received_img_pred, nodes_img_pred):')
        #         # print(timer_2x_no_addsig.timeit(100))
        #         # print()
        #
        #
        #         # timer_1x = Timer(
        #         #     stmt='self.get_new_nodes_img_pred_1x(message_received_img_pred, nodes_img_pred)',
        #         #     setup='',
        #         #     globals={
        #         #         'self': self,
        #         #         'message_received_img_pred': message_received_img_pred,
        #         #         'nodes_img_pred': nodes_img_pred,
        #         #     }
        #         # )
        #         # print('NEW 1x: get_new_nodes_img_pred_1x(message_received_img_pred, nodes_img_pred):')
        #         # print(timer_1x.timeit(100))
        #         # print()
        #
        #         # pred_old = torch_sigmoid(self.fc_eq3_w_img_pred(message_received_img_pred_old) + self.fc_eq3_u_img_pred(nodes_img_pred_old)) # torch.Size([506, 1024])
        #         # r_img_pred_old = torch_sigmoid(self.fc_eq4_w_img_pred(message_received_img_pred_old) + self.fc_eq4_u_img_pred(nodes_img_pred_old))
        #         # h_img_pred_old = torch_tanh(self.fc_eq5_w_img_pred(message_received_img_pred_old) + self.fc_eq5_u_img_pred(r_img_pred_old * nodes_img_pred_old))
        #         # # del message_received_img_pred, r_img_pred
        #         # nodes_img_pred_old = (1 - z_img_pred_old) * nodes_img_pred_old + z_img_pred_old * h_img_pred_old # nodes_img_pred: torch.Size([506, 1024])
        #         # import pdb; pdb.set_trace()
        #         # del z_img_pred, h_img_pred
        #         # end_old = time_time_ns()
        #         # time_old = end_old-start_old
        #         # print(f'OLD: 4 formulas: time_elapsed (ns)={time_old}')
        #
        #         # start_new = time_time_ns()
        #         r_img_pred = addsigmoid(self.fc_eq4_w_img_pred(message_received_img_pred), self.fc_eq4_u_img_pred(nodes_img_pred))
        #         nodes_img_pred = formula(nodes_img_pred,
        #             self.fc_eq3_w_img_pred(message_received_img_pred),
        #             self.fc_eq3_u_img_pred(nodes_img_pred),
        #             self.fc_eq5_w_img_pred(message_received_img_pred),
        #             self.fc_eq5_u_img_pred(r_img_pred * nodes_img_pred),
        #         )
        #         del r_img_pred
        #         # end_new = time_time_ns()
        #         # del r_img_pred
        #         # time_new = end_new-start_new
        #         # print(f'NEW: 2 jit functions: time_elapsed (ns)={time_new}')
        #         # print(f'NEW-OLD: for main formula: time_diff (ns)={time_new-time_old}. Negative is faster.')
        #         # if self.use_lrga is True:
        #         #     nodes_img_pred = self.dimension_reduce[t](torch_cat((self.attention[t](original_vr), nodes_img_pred), dim=1))
        #         #     if t != self.time_step_num - 1:
        #         #         # No ReLU nor batchnorm for last layer
        #         #         nodes_img_pred = self.gn[t](F_relu(nodes_img_pred))
        #
        #         # if not with_clean_classifier:
        #         # breakpoint()
        #         # time_old = Timer(
        #         #     stmt='torch_mm(self.fc_output_proj_img_pred(nodes_img_pred), self.fc_output_proj_ont_pred(nodes_ont_pred).t())',
        #         #     # setup='from __main__ import torch_mm',
        #         #     label='old torch_mm with second arg transpose',
        #         #     num_threads=num_threads,
        #         #     globals={
        #         #         'torch_mm': torch_mm,
        #         #         'self': self,
        #         #         'nodes_img_pred': nodes_img_pred,
        #         #         'nodes_ont_pred': nodes_ont_pred,
        #         #     }
        #         # )
        #         #
        #         # print('OLD: torch_mm(a, bT):')
        #         # print(time_old.timeit(100))
        #         # print(f'OLD: torch_mm(a, bT): time_elapsed={time_old.timeit(100)}')
        #
        #         # start_new = time_time_ns()
        #         # pred_cls_logits = torch_mm_bT(self.fc_output_proj_img_pred(nodes_img_pred), self.fc_output_proj_ont_pred(nodes_ont_pred)) # torch.Size([506, 1024]) @ torch.Size([1024, 51])
        #         # end_new = time_time_ns()
        #         # timer_new = Timer(
        #         #     stmt='torch_mm_bT(self.fc_output_proj_img_pred(nodes_img_pred), self.fc_output_proj_ont_pred(nodes_ont_pred))',
        #         #     # setup='from __main__ import torch_mm_bT',
        #         #     num_threads=num_threads,
        #         #     label='jit torch_mm with second arg transpose',
        #         #     globals={
        #         #         'torch_mm_bT': torch_mm_bT,
        #         #         'self': self,
        #         #         'nodes_img_pred': nodes_img_pred,
        #         #         'nodes_ont_pred': nodes_ont_pred,
        #         #     }
        #         # )
        #         # time_new = end_new-start_new
        #         # print('NEW: torch_mm_bT(a, b):')
        #         # print(timer_new.timeit(100))
        #
        #         # start_old = time_time_ns()
        #         pred_cls_logits = torch_mm(self.fc_output_proj_img_pred(nodes_img_pred), self.fc_output_proj_ont_pred(nodes_ont_pred).t()) # torch.Size([506, 1024]) @ torch.Size([1024, 51])
        #         # end_old = time_time_ns()
        #         # time_old = end_old-start_old
        #
        #         # print(f'NEW: torch_mm_bT(a, b): time_elapsed={timer_new.timeit(100)}')
        #         # print(f'NEW-OLD: for torch_mm_bT: time_diff (ns)={time_new-time_old}. Negative is faster.')
        #         # if with_clean_classifier:
        #         #     pred_cls_logits = torch_mm(self.fc_output_proj_img_pred_clean(nodes_img_pred), self.fc_output_proj_ont_pred_clean(nodes_ont_pred).t())
        #         # if self.sa is True and self.merge_eoa_sa is True:
        #         #     pred_cls_logits = (F_normalize(pred_adj_nor + self.ontological_preds, p=1) @ pred_cls_logits.T).T
        #         # if self.sa is True and self.merge_eoa_sa is False:
        #         #     pred_cls_logits = (pred_adj_nor @ pred_cls_logits.T).T
        #         #
        #         # if self.use_ontological_adjustment is True and not self.merge_eoa_sa:
        #         #     pred_cls_logits = (self.ontological_preds @ pred_cls_logits.T).T
        #
        #         # import pdb; pdnodes_img_pred_oldb.set_trace()
        #         edges_img2ont_pred = F_softmax(pred_cls_logits, dim=1)
        #         edges_ont2img_pred = edges_img2ont_pred.t()
        #         if self.refine_obj_cls:
        #             ent_cls_logits = torch_mm(self.fc_output_proj_img_ent(nodes_img_ent), self.fc_output_proj_ont_ent(nodes_ont_ent).t())
        #             edges_img2ont_ent = F_softmax(ent_cls_logits, dim=1)
        #             edges_ont2img_ent = edges_img2ont_ent.t()
        #         else:
        #             ent_cls_logits = obj_dists
        #         # breakpoint()
        #
        #     ent_cls_logits_all.append(ent_cls_logits)
        #     pred_cls_logits_all.append(pred_cls_logits)
        #     # debug_memory()
        #     compare = Compare(results)
        #     compare.print()
        #     # debug_memory()
        #     # print(gpu_profile(frame=_getframe(), event='line', arg=None))
        #     del results, compare
        # # breakpoint()
        # return ent_cls_logits_all, pred_cls_logits_all, {}

    def per_image(self, proposal, rel_pair_idx, roi_features_image, union_features_image, device=None, debug=True):
        obj_labels = proposal.get_field("labels")

        if self.mode == 'predcls':
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
        # breakpoint()
        # if self.mode == 'predcls':
        #     obj_preds = to_onehot(obj_preds, self.num_obj_cls)

        num_objs = len(proposal) #0: 15
        num_rels = len(rel_pair_idx) #0: 210

        nodes_ont_ent = self.fc_init_ont_ent(torch_as_tensor(self.emb_ent, dtype=torch_float32, device=device))
        nodes_ont_pred = self.fc_init_ont_pred(torch_as_tensor(self.emb_pred, dtype=torch_float32, device=device))
        # nodes_img_ent = roi_features[idx_start_ent:idx_start_ent+num_objs]
        nodes_img_ent = roi_features_image
        # idx_start_ent += num_objs
        # nodes_img_pred = union_features[idx_start_pred:idx_start_pred+num_rels]
        nodes_img_pred = union_features_image
        # idx_start_pred += num_rels

        # TODO: use placeholders instead of generating new tensors which
        # require a recreation of the computation graph. This may be true
        # even if requires_grad = False.
        edges_ont_ent2ent = torch_as_tensor(self.adjmtx_ent2ent, dtype=torch_float32, device=device)
        edges_ont_ent2pred = torch_as_tensor(self.adjmtx_ent2pred, dtype=torch_float32, device=device)
        edges_ont_pred2ent = torch_as_tensor(self.adjmtx_pred2ent, dtype=torch_float32, device=device)
        edges_ont_pred2pred = torch_as_tensor(self.adjmtx_pred2pred, dtype=torch_float32, device=device)

        edges_img_pred2subj = torch_zeros((num_rels, num_objs), dtype=torch_float32, device=device)
        edges_img_pred2subj[:, rel_pair_idx[:, 0]] = 1
        edges_img_pred2obj = torch_zeros((num_rels, num_objs), dtype=torch_float32, device=device)
        edges_img_pred2obj[:, rel_pair_idx[:, 1]] = 1
        edges_img_subj2pred = edges_img_pred2subj.t() # TODO: need to specify axes when vectorized
        edges_img_obj2pred = edges_img_pred2obj.t()

        edges_img2ont_ent = obj_dists.clone().detach()
        edges_ont2img_ent = edges_img2ont_ent.t()

        num_rel_cls = self.num_rel_cls
        edges_img2ont_pred = torch_zeros((num_rels, num_rel_cls), dtype=torch_float32, device=device, requires_grad=False) # torch.Size([506, 51])
        edges_ont2img_pred = edges_img2ont_pred.t()

        ent_cls_logits = None

        num_edge_types_ent2ent = self.num_edge_types_ent2ent
        num_edge_types_pred2ent = self.num_edge_types_pred2ent
        num_edge_types_ent2pred = self.num_edge_types_ent2pred
        num_edge_types_pred2pred = self.num_edge_types_pred2pred

        num_threads = torch_get_num_threads()
        print(f'Benchmarking on {num_threads} threads')

        results = []

        # with_clean_classifier = self.with_clean_classifier

        for t in range(self.time_step_num):
            # breakpoint()
            message_send_ont_ent = self.fc_mp_send_ont_ent(nodes_ont_ent) # torch.Size([177, 1024]) => torch.Size([177, 256])
            # breakpoint()
            message_send_ont_pred = self.fc_mp_send_ont_pred(nodes_ont_pred) # torch.Size([51, 1024]) => torch.Size([51, 256])
            # breakpoint()
            message_send_img_ent = self.fc_mp_send_img_ent(nodes_img_ent) # torch.Size([23, 1024]) => torch.Size([23, 256])
            # breakpoint()
            message_send_img_pred = self.fc_mp_send_img_pred(nodes_img_pred) # torch.Size([506, 1024]) => torch.Size([506, 256])
            # breakpoint()
            message_received_ont_ent = torch_cat(
                [torch_mm(edges_ont_ent2ent[i].t(), message_send_ont_ent) for i in range(num_edge_types_ent2ent)] + # 177, 177 => 177, 256 x 9
                [torch_mm(edges_ont_pred2ent[i].t(), message_send_ont_pred) for i in range(num_edge_types_pred2ent)] + # 51 177 => 51, 256 * 3
                [torch_mm(edges_img2ont_ent.t(), message_send_img_ent),]
            , 1) #  torch.Size([177, 3328])
            # NOTE: there's some vectorization opportunity right here.
            # breakpoint()
            message_received_ont_ent = self.fc_mp_receive_ont_ent(message_received_ont_ent) # message_received_ont_ent: torch.Size([177, 3328]) => torch.Size([177, 1024])

            # edges_ont_ent2pred: torch.Size([3, 151, 51])
            # message_send_ont_ent: torch.Size([151, 256])
            # og = torch_cat([torch_mm(edges_ont_ent2pred[i].t(), message_send_ont_ent) for i in range(num_edge_types_ent2pred)], 1)
            # # og: 51, 768
            # breakpoint()
            # lol = torch_einsum('abc,bd->cad', edges_ont_ent2pred, message_send_ont_ent).view(num_rel_cls, -1) # 3, 51, 256

            # import torch
            # # b=3, n=151, m=51; b=1, m=51, p=256; x
            # # b=51, n=3, m=151, b=51, m=151, p=1; v
            # lol = edges_ont_ent2pred.view(3, 151, 51)
            # message_send_ont_pred.unsqueeze(1)
            # lol = torch.movedim(edges_ont_ent2pred, 2, 0)
            # TODO:
            # [a=3, b=151, c=51], [151, 256] => [3, 51, 256]
            # a = torch.rand(3,151,51)
            # b = torch.rand(151,256)
            # c = torch.cat([torch.mm(a[zi].t(), b) for i in range(3)], 1)
            # c1 = torch.mm(a[0].t(), b)
            #
            # lol = torch.einsum('abc,bd->cad', edges_ont_ent2pred, message_send_ont_ent)
            # # c2 = torch.einsum('abc,bd->acd', a, b)
            # c2 = torch.einsum('abc,bd->cad', a, b)
            # c2v = c2.view(51, 768)
            # # c1n = c2[0, :, :]
            # # cn = c2.permute(1, 0, 2).reshape(51, 768)
            # # torch.equal(c1, c1n)
            # # torch.equal(c, cn)
            # lol: torch.Size([3, 51, 256])

            # [a=3, b=151, c=51], [151, 256] => [3, 51, 256]
            # torch.einsum("abc,bd->acd")
            # lol = torch.movedim(edges_ont_ent2pred, 2, 0) # not contiguous
            # torch.bmm(edges_ont_ent2pred.moveaxis(), )
            # torch.bmm(message_send_ont_ent.unsqueeze(-1))
            # ijk = [3, 151, 51]; [51, 256]
            # i = 3, j = 151, k = 51,
            # jk
            # lol = torch.einsum('ijk, kl -> ikl', edges_ont_ent2pred, message_send_ont_ent)
            # lol = torch.einsum('ijk, nkl -> kli', edges_ont_ent2pred.view(51, ), message_send_ont_ent.unsqueeze(0))
            # edges_ont_ent2pred: 3, 151, 51
            # edges_ont_ent2pred[0] # 151, 51
            # message_send_ont_ent: 151, 126
            # torch_mm(edges_ont_ent2pred[i].t(), message_send_ont_ent) # torch.Size([51, 256])
            # new = torch_mm(edges_ont_ent2pred, message_send_ont_ent)
            message_received_ont_pred = torch_cat(
                [torch_mm(edges_ont_ent2pred[i].t(), message_send_ont_ent) for i in range(num_edge_types_ent2pred)] +
                [torch_mm(edges_ont_pred2pred[i].t(), message_send_ont_pred) for i in range(num_edge_types_pred2pred)] +
                [torch_mm(edges_img2ont_pred.t(), message_send_img_pred),] # edges_ont2img_pred: torch.Size([51, 506]) @ torch.Size([506, 256])
            , 1) # torch.Size([51, 2048]) # Key is here. Why is this set?
            message_received_ont_pred = self.fc_mp_receive_ont_pred(message_received_ont_pred) # torch.Size([51, 2048]) => torch.Size([51, 1024])

            # breakpoint()
            message_received_img_ent = torch_cat([
                torch_mm(edges_img_pred2subj.t(), message_send_img_pred),
                torch_mm(edges_img_pred2obj.t(), message_send_img_pred),
                torch_mm(edges_ont2img_ent.t(), message_send_ont_ent),
            ], 1) # torch.Size([23, 768])
            message_received_img_ent = self.fc_mp_receive_img_ent(message_received_img_ent) # torch.Size([23, 768]) => torch.Size([23, 1024])
            # import pdb; pdb.set_trace()

            del message_send_ont_ent, message_send_img_pred

            # breakpoint()
            message_received_img_pred = torch_cat([
                torch_mm(edges_img_subj2pred.t(), message_send_img_ent),
                torch_mm(edges_img_obj2pred.t(), message_send_img_ent),
                torch_mm(edges_ont2img_pred.t(), message_send_ont_pred),
            ], 1) # torch.Size([with_clean_classifier506, 768])
            message_received_img_pred = self.fc_mp_receive_img_pred(message_received_img_pred) # torch.Size([506, 768])=>torch.Size([506, 1024])

            del message_send_ont_pred, message_send_img_ent

            r_ont_ent = addsigmoid(self.fc_eq4_w_ont_ent(message_received_ont_ent), self.fc_eq4_u_ont_ent(nodes_ont_ent))
            nodes_ont_ent = formula(nodes_ont_ent,
                self.fc_eq3_w_ont_ent(message_received_ont_ent),
                self.fc_eq3_u_ont_ent(nodes_ont_ent),
                self.fc_eq5_w_ont_ent(message_received_ont_ent),
                self.fc_eq5_u_ont_ent(r_ont_ent * nodes_ont_ent),
            )
            del r_ont_ent

            r_ont_pred = addsigmoid(self.fc_eq4_w_ont_pred(message_received_ont_pred), self.fc_eq4_u_ont_pred(nodes_ont_pred))
            nodes_ont_pred = formula(nodes_ont_pred,
                self.fc_eq3_w_ont_pred(message_received_ont_pred),
                self.fc_eq3_u_ont_pred(nodes_ont_pred),
                self.fc_eq5_w_ont_pred(message_received_ont_pred),
                self.fc_eq5_u_ont_pred(r_ont_pred * nodes_ont_pred),
            )
            del r_ont_pred

            r_img_ent = addsigmoid(self.fc_eq4_w_img_ent(message_received_img_ent), self.fc_eq4_u_img_ent(nodes_img_ent))
            nodes_img_ent = formula(nodes_img_ent,
                self.fc_eq3_w_img_ent(message_received_img_ent),
                self.fc_eq3_u_img_ent(nodes_img_ent),
                self.fc_eq5_w_img_ent(message_received_img_ent),
                self.fc_eq5_u_img_ent(r_img_ent * nodes_img_ent),
            )
            del r_img_ent

            # nodes_img_pred_old = nodes_img_pred.clone().detach()
            # message_received_img_pred_old = message_received_img_pred.clone().detach()
            # start_old = time_time_ns()
            # debug_memory()
            size = message_received_img_pred.size()
            timer_old = Timer(
                label='gating formula',
                sub_label=f'{size}_{t}',
                description='old',
                stmt='lol = self.get_new_nodes_img_pred_old(message_received_img_pred, nodes_img_pred)',
                setup='',
                globals={
                    'self': self,
                    'message_received_img_pred': message_received_img_pred,
                    'nodes_img_pred': nodes_img_pred,
                }
            ).blocked_autorange(min_run_time=1)
            results.append(timer_old)
            # print('OLD: 4x get_new_nodes_img_pred_old(message_received_img_pred, nodes_img_pred):')
            # print(timer_old.timeit(100))
            # print()
            # print(gpu_profile(frame=_getframe(), event='line', arg=None))
            # print('timer_old:')
            # print(print_para(self))
            # debug_memory()

            timer_2x = Timer(
                label='gating formula',
                sub_label=f'{size}_{t}',
                description='2x',
                stmt='lol = self.get_new_nodes_img_pred_2x(message_received_img_pred, nodes_img_pred)',
                setup='',
                globals={
                    'self': self,
                    'message_received_img_pred': message_received_img_pred,
                    'nodes_img_pred': nodes_img_pred,
                }
            ).blocked_autorange(min_run_time=1)
            results.append(timer_2x)
            # print('timer_2x:')
            # print(print_para(self))
            # debug_memory()
            # print('NEW 2x: get_new_nodes_img_pred_2x(message_received_img_pred, nodes_img_pred):')
            # print(timer_2x.timeit(100))
            # print()
            # print(gpu_profile(frame=_getframe(), event='line', arg=None))

            timer_2x_no_addsig = Timer(
                label='gating formula',
                sub_label=f'{size}_{t}',
                description='2x_no_addsig',
                stmt='lol = self.get_new_nodes_img_pred_2x_no_addsigmoid(message_received_img_pred, nodes_img_pred)',
                setup='',
                globals={
                    'self': self,
                    'message_received_img_pred': message_received_img_pred,
                    'nodes_img_pred': nodes_img_pred,
                }
            ).blocked_autorange(min_run_time=1)
            results.append(timer_2x_no_addsig)
            print('timer_2x_no_addsig:')
            # print(print_para(self))
            # debug_memory()
            # print(gpu_profile(frame=_getframe(), event='line', arg=None))

            timer_2x_no_addsig_oneline = Timer(
                label='gating formula',
                sub_label=f'{size}_{t}',
                description='2x_no_addsig_oneline',
                stmt='lol = self.get_new_nodes_img_pred_2x_no_addsigmoid_oneline(message_received_img_pred, nodes_img_pred)',
                setup='',
                globals={
                    'self': self,
                    'message_received_img_pred': message_received_img_pred,
                    'nodes_img_pred': nodes_img_pred,
                }
            ).blocked_autorange(min_run_time=1)
            results.append(timer_2x_no_addsig_oneline)
            print('timer_2x_no_addsig_oneline:')
            # print(print_para(self))
            # debug_memory()
            # print(gpu_profile(frame=_getframe(), event='line', arg=None))

            timer_2x_no_addsig_oneline_out_cls = Timer(
                label='gating formula',
                sub_label=f'{size}_{t}',
                description='2x_no_addsig_oneline_out_cls',
                stmt='lol = self.get_new_nodes_img_pred_2x_no_addsigmoid_oneline_out_cls(message_received_img_pred, nodes_img_pred)',
                setup='',
                globals={
                    'self': self,
                    'message_received_img_pred': message_received_img_pred,
                    'nodes_img_pred': nodes_img_pred,
                }
            ).blocked_autorange(min_run_time=1)
            results.append(timer_2x_no_addsig_oneline_out_cls)
            print('timer_2x_no_addsig_oneline_out_cls:')
            # print(print_para(self))
            # debug_memory()
            # print(gpu_profile(frame=_getframe(), event='line', arg=None))

            # timer_2x_no_addsig_oneline_out_fn = Timer(
            #     label='gating formula',
            #     sub_label=f'{size}_{t}',
            #     description='2x_no_addsig_oneline_out_fn',
            #     stmt='self.get_new_nodes_img_pred_2x_no_addsigmoid_oneline_out_fn(message_received_img_pred, nodes_img_pred)',
            #     setup='',
            #     globals={
            #         'self': self,
            #         'message_received_img_pred': message_received_img_pred,
            #         'nodes_img_pred': nodes_img_pred,
            #     }
            # ).blocked_autorange(min_run_time=1)
            # results.append(ti Timer(
            #     label='gating formula',
            #     sub_label=f'{size}_{t}',
            #     description='2x_no_addsig_oneline_out_fn',
            #     stmt='self.get_new_nodes_img_pred_2x_no_addsigmoid_oneline_out_fn(message_received_img_pred, nodes_img_pred)',
            #     setup='',
            #     globals={
            #         'self': self,
            #         'message_received_img_pred': message_received_img_pred,
            #         'nodes_img_pred': nodes_img_pred,
            #     }
            # ).blocked_autorange(min_run_time=1)
            # results.append(timemer_2x_no_addsig_oneline_out_fn)
            # print('NEW 2x_no_addsig: get_new_nodes_img_pred_2x_no_addsigmoid(message_received_img_pred, nodes_img_pred):')
            # print(timer_2x_no_addsig.timeit(100))
            # print()


            # timer_1x = Timer(
            #     stmt='self.get_new_nodes_img_pred_1x(message_received_img_pred, nodes_img_pred)',
            #     setup='',
            #     globals={
            #         'self': self,
            #         'message_received_img_pred': message_received_img_pred,
            #         'nodes_img_pred': nodes_img_pred,
            #     }
            # )
            # print('NEW 1x: get_new_nodes_img_pred_1x(message_received_img_pred, nodes_img_pred):')
            # print(timer_1x.timeit(100))
            # print()

            # pred_old = torch_sigmoid(self.fc_eq3_w_img_pred(message_received_img_pred_old) + self.fc_eq3_u_img_pred(nodes_img_pred_old)) # torch.Size([506, 1024])
            # r_img_pred_old = torch_sigmoid(self.fc_eq4_w_img_pred(message_received_img_pred_old) + self.fc_eq4_u_img_pred(nodes_img_pred_old))
            # h_img_pred_old = torch_tanh(self.fc_eq5_w_img_pred(message_received_img_pred_old) + self.fc_eq5_u_img_pred(r_img_pred_old * nodes_img_pred_old))
            # # del message_received_img_pred, r_img_pred
            # nodes_img_pred_old = (1 - z_img_pred_old) * nodes_img_pred_old + z_img_pred_old * h_img_pred_old # nodes_img_pred: torch.Size([506, 1024])
            # import pdb; pdb.set_trace()
            # del z_img_pred, h_img_pred
            # end_old = time_time_ns()
            # time_old = end_old-start_old
            # print(f'OLD: 4 formulas: time_elapsed (ns)={time_old}')

            # start_new = time_time_ns()
            r_img_pred = addsigmoid(self.fc_eq4_w_img_pred(message_received_img_pred), self.fc_eq4_u_img_pred(nodes_img_pred))
            nodes_img_pred = formula(nodes_img_pred,
                self.fc_eq3_w_img_pred(message_received_img_pred),
                self.fc_eq3_u_img_pred(nodes_img_pred),
                self.fc_eq5_w_img_pred(message_received_img_pred),
                self.fc_eq5_u_img_pred(r_img_pred * nodes_img_pred),
            )
            del r_img_pred
            # end_new = time_time_ns()
            # del r_img_pred
            # time_new = end_new-start_new
            # print(f'NEW: 2 jit functions: time_elapsed (ns)={time_new}')
            # print(f'NEW-OLD: for main formula: time_diff (ns)={time_new-time_old}. Negative is faster.')
            # if self.use_lrga is True:
            #     nodes_img_pred = self.dimension_reduce[t](torch_cat((self.attention[t](original_vr), nodes_img_pred), dim=1))
            #     if t != self.time_step_num - 1:
            #         # No ReLU nor batchnorm for last layer
            #         nodes_img_pred = self.gn[t](F_relu(nodes_img_pred))

            # if not with_clean_classifier:
            # breakpoint()
            # time_old = Timer(
            #     stmt='torch_mm(self.fc_output_proj_img_pred(nodes_img_pred), self.fc_output_proj_ont_pred(nodes_ont_pred).t())',
            #     # setup='from __main__ import torch_mm',
            #     label='old torch_mm with second arg transpose',
            #     num_threads=num_threads,
            #     globals={
            #         'torch_mm': torch_mm,
            #         'self': self,
            #         'nodes_img_pred': nodes_img_pred,
            #         'nodes_ont_pred': nodes_ont_pred,
            #     }
            # )
            #
            # print('OLD: torch_mm(a, bT):')
            # print(time_old.timeit(100))
            # print(f'OLD: torch_mm(a, bT): time_elapsed={time_old.timeit(100)}')

            # start_new = time_time_ns()
            # pred_cls_logits = torch_mm_bT(self.fc_output_proj_img_pred(nodes_img_pred), self.fc_output_proj_ont_pred(nodes_ont_pred)) # torch.Size([506, 1024]) @ torch.Size([1024, 51])
            # end_new = time_time_ns()
            # timer_new = Timer(
            #     stmt='torch_mm_bT(self.fc_output_proj_img_pred(nodes_img_pred), self.fc_output_proj_ont_pred(nodes_ont_pred))',
            #     # setup='from __main__ import torch_mm_bT',
            #     num_threads=num_threads,
            #     label='jit torch_mm with second arg transpose',
            #     globals={
            #         'torch_mm_bT': torch_mm_bT,
            #         'self': self,
            #         'nodes_img_pred': nodes_img_pred,
            #         'nodes_ont_pred': nodes_ont_pred,
            #     }
            # )
            # time_new = end_new-start_new
            # print('NEW: torch_mm_bT(a, b):')
            # print(timer_new.timeit(100))

            # start_old = time_time_ns()
            pred_cls_logits = torch_mm(self.fc_output_proj_img_pred(nodes_img_pred), self.fc_output_proj_ont_pred(nodes_ont_pred).t()) # torch.Size([506, 1024]) @ torch.Size([1024, 51])
            # end_old = time_time_ns()
            # time_old = end_old-start_old

            # print(f'NEW: torch_mm_bT(a, b): time_elapsed={timer_new.timeit(100)}')
            # print(f'NEW-OLD: for torch_mm_bT: time_diff (ns)={time_new-time_old}. Negative is faster.')
            # if with_clean_classifier:
            #     pred_cls_logits = torch_mm(self.fc_output_proj_img_pred_clean(nodes_img_pred), self.fc_output_proj_ont_pred_clean(nodes_ont_pred).t())
            # if self.sa is True and self.merge_eoa_sa is True:
            #     pred_cls_logits = (F_normalize(pred_adj_nor + self.ontological_preds, p=1) @ pred_cls_logits.T).T
            # if self.sa is True and self.merge_eoa_sa is False:
            #     pred_cls_logits = (pred_adj_nor @ pred_cls_logits.T).T
            #
            # if self.use_ontological_adjustment is True and not self.merge_eoa_sa:
            #     pred_cls_logits = (self.ontological_preds @ pred_cls_logits.T).T

            # import pdb; pdnodes_img_pred_oldb.set_trace()
            edges_img2ont_pred = F_softmax(pred_cls_logits, dim=1)
            edges_ont2img_pred = edges_img2ont_pred.t()
            if self.refine_obj_cls:
                ent_cls_logits = torch_mm(self.fc_output_proj_img_ent(nodes_img_ent), self.fc_output_proj_ont_ent(nodes_ont_ent).t())
                edges_img2ont_ent = F_softmax(ent_cls_logits, dim=1)
                edges_ont2img_ent = edges_img2ont_ent.t()
            else:
                ent_cls_logits = obj_dists
            # breakpoint()

        # debug_memory()
        compare = Compare(results)
        compare.print()
        # debug_memory()
        # print(gpu_profile(frame=_getframe(), event='line', arg=None))
        del results, compare
        return ent_cls_logits, pred_cls_logits

# @torch_jit_script
# def addtmul(x, a, b):
#     return (1 - a) * x + a * b

    def get_new_nodes_img_pred_old(self, message_received_img_pred, nodes_img_pred):
        z_img_pred = torch_sigmoid(self.fc_eq3_w_img_pred(message_received_img_pred) + self.fc_eq3_u_img_pred(nodes_img_pred)) # torch.Size([506, 1024])
        r_img_pred = torch_sigmoid(self.fc_eq4_w_img_pred(message_received_img_pred) + self.fc_eq4_u_img_pred(nodes_img_pred))
        h_img_pred = torch_tanh(self.fc_eq5_w_img_pred(message_received_img_pred) + self.fc_eq5_u_img_pred(r_img_pred * nodes_img_pred))
        # del message_received_img_pred, r_img_pred
        return (1 - z_img_pred) * nodes_img_pred + z_img_pred * h_img_pred # nodes_img_pred: torch.Size([506, 1024])

    def get_new_nodes_img_pred_2x(self, message_received_img_pred, nodes_img_pred):
        r_img_pred = addsigmoid(self.fc_eq4_w_img_pred(message_received_img_pred), self.fc_eq4_u_img_pred(nodes_img_pred))
        return formula(nodes_img_pred,
            self.fc_eq3_w_img_pred(message_received_img_pred),
            self.fc_eq3_u_img_pred(nodes_img_pred),
            self.fc_eq5_w_img_pred(message_received_img_pred),
            self.fc_eq5_u_img_pred(r_img_pred * nodes_img_pred),
        )

    def get_new_nodes_img_pred_2x_no_addsigmoid(self, message_received_img_pred, nodes_img_pred):
        r_img_pred = torch_sigmoid(self.fc_eq4_w_img_pred(message_received_img_pred) + self.fc_eq4_u_img_pred(nodes_img_pred))
        return formula(nodes_img_pred,
            self.fc_eq3_w_img_pred(message_received_img_pred),
            self.fc_eq3_u_img_pred(nodes_img_pred),
            self.fc_eq5_w_img_pred(message_received_img_pred),
            self.fc_eq5_u_img_pred(r_img_pred * nodes_img_pred),
        )

    def get_new_nodes_img_pred_2x_no_addsigmoid_oneline(self, message_received_img_pred, nodes_img_pred):
        r_img_pred = torch_sigmoid(self.fc_eq4_w_img_pred(message_received_img_pred) + self.fc_eq4_u_img_pred(nodes_img_pred))
        return formula_oneline(nodes_img_pred,
            self.fc_eq3_w_img_pred(message_received_img_pred),
            self.fc_eq3_u_img_pred(nodes_img_pred),
            self.fc_eq5_w_img_pred(message_received_img_pred),
            self.fc_eq5_u_img_pred(r_img_pred * nodes_img_pred),
        )

    def get_new_nodes_img_pred_2x_no_addsigmoid_oneline_out_cls(self, message_received_img_pred, nodes_img_pred):
        r_img_pred = torch_sigmoid(self.fc_eq4_w_img_pred(message_received_img_pred) + self.fc_eq4_u_img_pred(nodes_img_pred))
        return formula_oneline_out_cls(nodes_img_pred,
            self.fc_eq3_w_img_pred(message_received_img_pred),
            self.fc_eq3_u_img_pred(nodes_img_pred),
            self.fc_eq5_w_img_pred(message_received_img_pred),
            self.fc_eq5_u_img_pred(r_img_pred * nodes_img_pred),
        )

    # NOTE: message_received_img_pred and nodes_img_pred requires grad so can't use out=.
    # def get_new_nodes_img_pred_2x_no_addsigmoid_oneline_out_fn(self, message_received_img_pred, nodes_img_pred):
    #     r_img_pred = torch_sigmoid(self.fc_eq4_w_img_pred(message_received_img_pred) + self.fc_eq4_u_img_pred(nodes_img_pred))
    #     return formula_oneline_out_fn(nodes_img_pred,
    #         self.fc_eq3_w_img_pred(message_received_img_pred),
    #         self.fc_eq3_u_img_pred(nodes_img_pred),
    #         self.fc_eq5_w_img_pred(message_received_img_pred),
    #         self.fc_eq5_u_img_pred(r_img_pred * nodes_img_pred),
    #     )


    # def get_new_nodes_img_pred_1x(self, message_received_img_pred, nodes_img_pred):
    #     return get_new_nodes_img_pred_1x(
    #         self.fc_eq3_w_img_pred(message_received_img_pred), # eq3a
    #         self.fc_eq3_u_img_pred(nodes_img_pred), # eq3b
    #         self.fc_eq4_w_img_pred(message_received_img_pred), # eq4a
    #         self.fc_eq4_u_img_pred(nodes_img_pred), # eq4b
    #         self.fc_eq5_w_img_pred(message_received_img_pred), # eq5a
    #         self.fc_eq5_u_img_pred,
    #         nodes_img_pred,
    #     )

# @torch_jit_script
# def get_new_nodes_img_pred_1x(eq3a, eq3b, eq4a, eq4b, eq5a, eq5, b):
#     # z_img_pred = torch_sigmoid(self.fc_eq3_w_img_pred(message_received_img_pred) + self.fc_eq3_u_img_pred(nodes_img_pred)) # torch.Size([506, 1024])
#     z_img_pred = torch_sigmoid(eq3a + eq3b) # torch.Size([506, 1024])
#     # del message_received_img_pred, r_img_pred
#     return (1 - z_img_pred) * b + z_img_pred * torch_tanh(eq5a + eq5(torch_sigmoid(eq4a + eq4b) * b))

# NOTE: Old is faster
# @torch_jit_script
# def torch_mm_bT(a, b):
#     return torch_mm(a, b.T)


@torch_jit_script
def addsigmoid(a, b):
    return torch_sigmoid(a + b)

# @torch_jit_script
# def addtahn(a, b):
#     return torch_tanh(a + b)

@torch_jit_script
def formula(x, a, b, e, f):
    z = torch_sigmoid(a + b) # torch.Size([23, 1024])
    # r_img_ent = torch_sigmoid(c + d)
    del a, b
    h = torch_tanh(e + f)
    del e, f
    return (1 - z) * x + z * h # nodes_img_ent: torch.Size([23, 1024])

@torch_jit_script
def formula_oneline(x, a, b, e, f):
    z = torch_sigmoid(a + b) # torch.Size([23, 1024])
    del a, b
    # r_img_ent = torch_sigmoid(c + d)
    return (1 - z) * x + z * torch_tanh(e + f) # nodes_img_ent: torch.Size([23, 1024])

@torch_jit_script
def formula_oneline_out_cls(x, a, b, e, f):
    a.add_(b).sigmoid_() # torch.Size([23, 1024])
    del b
    # r_img_ent = torch_sigmoid(c + d)
    # x.mul_((1 - a)).add_(e.add_(f).tanh_().mul_(a)) # nodes_img_ent: torch.Size([23, 1024])
    return x.mul((1 - a)).add_(e.add_(f).tanh_().mul_(a)) # nodes_img_ent: torch.Size([23, 1024])
    # del a, e, f
# @torch_jit_script
# def formula_oneline_out_fn(x, a, b, e, f):
#     torch_sigmoid(torch_add(a, b, out=a), out=a) # torch.Size([23, 1024])
#     # r_img_ent = torch_sigmoid(c + d)
#     torch_add(torch_mul((1 - a), x, out=x), torch_mul(a, torch_tanh(torch_add(e, f, out=e), out=e), out=e), out=x) # nodes_img_ent: torch.Size([23, 1024])
    # nodes_img_pred: torch.Size([506, 1024])
            # TODO: need to do a few continuous calls maybe in the matrix ops

            # if self.mode == 'predcls':
            #     obj_preds = to_onehot(obj_preds, self.num_obj_cls)

        # import pdb; pdb.set_trace()
        # use_gt_label = self.training or self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        # obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        #
        # if obj_labels is not None:
        #     obj_labels = obj_labels.long()
        # # label/logits embedding will be used as input
        # if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
        #     obj_embed = self.obj_embed1(obj_labels)
        # else:
        #     obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
        #     obj_embed = F_softmax(obj_logits, dim=1) @ self.obj_embed1.weight
        # # if self.attribute_on:
        # #     obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        # # else:
        # #     obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)
        # num_objs = [len(p) for p in proposals]
        #
        # # predict obj_dists and obj_preds
        # if self.mode == 'predcls':
        #     obj_preds = obj_labels
        #     obj_dists = to_onehot(obj_preds, self.num_obj_cls)
        #     # (Pdb) obj_dists.size()
        #     # torch.Size([1280, 151])
        #     # edge_pre_rep = cat((roi_features, obj_feats, features, self.obj_embed2(obj_labels)), dim=-1)
        #     # edge_pre_rep = cat((roi_features, obj_feats, self.obj_embed2(obj_labels)), dim=-1)
        # else:
        #     obj_feats = None # TODO: remove
        #     obj_dists = self.out_obj(obj_feats)
        #     # (Pdb) obj_dists.size()
        #     # torch.Size([1280, 151])
        #     use_decoder_nms = self.mode == 'sgdet' and not self.training
        #     if use_decoder_nms:
        #         boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
        #         obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
        #     else:
        #         obj_preds = obj_dists[:, 1:].max(1)[1] + 1
        #         # obj_preds are object predictions.
        #         # (Pdb) obj_preds.size()
        #         # torch.Size([1280])
        #         # What is obj_preds?
        # # post decode
        # # (Pdb) edge_ctx.size()
        # # torch.Size([1280, 768])
        # # edge_rep = self.post_emb(edge_ctx)
        # # # (Pdb) edge_ctx.size()
        # # # torch.Size([1280, 1536])
        # # edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim) # torch.Size([1280, 2, 768])
        # #
        # # head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim) # torch.Size([1280, 768])
        # # tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim) # torch.Size([1280, 768])
        # #
        # # num_rels = [r.shape[0] for r in rel_pair_idxs] # num_rels per image
        # # num_objs = [len(b) for b in proposals]
        # # assert len(num_rels) == len(num_objs)
        #
        # images = []
        # for image_idx, image in enumerate(zip(images, rel_pair_idxs, obj_preds)):
        #     rel_inds = rel_pair_idxs[image_idx] # vectorized rel_inds is just rel_pair_idxs
        #     obj_probs = F_softmax(obj_dists, 1) # already vectorized originally, or obj_logits
        #     obj_fmaps = roi_features # [] => [torch.Size([1280, 4096])]
        #     obj_fmaps = self.obj_proj(obj_fmaps)
        #     vr = self.rel_proj(vr)
        #     result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
        #                            train_anchor_inds, return_fmap=True)
        #     rois = torch_cat((im_inds[:, None].float(), boxes), 1)
        #
        #     vr = self.visual_rep(result.fmap.detach(), rois, rel_inds[:, 1:]) # union features? # Classify the features using non-pretrained backend.
        #
        #
        # # 1. vectorize the gbnet code.
        # # 2. bring it here
        # # head_reps = head_rep.split(num_objs, dim=0) # 16 * torch.Size([80, 768])
        # # tail_reps = tail_rep.split(num_objs, dim=0) # 16 * torch.Size([80, 768])
        # # obj_preds = obj_preds.split(num_objs, dim=0) # 16 * torch.Size([80])
        #
        # # from object level feature to pairwise relation level feature
        # prod_reps = []
        # pair_preds = []
        # pairs_culsum = 0
        # # num_rels_per_image = []
        # global_features_mapped = torch_empty_like(union_features, dtype=union_features.dtype, device=union_features.device)
        # new_to_old = torch_empty(union_features.size(0), dtype=int)
        # for img_idx, (pair_idx, head_rep, tail_rep, obj_pred) in enumerate(zip(rel_pair_idxs, head_reps, tail_reps, obj_preds)):
        #     prod_reps.append(torch_cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
        #     pair_preds.append(obj_pred[pair_idx])
        #     # num_rels_per_image.append(len(pair_idx))
        #     new_to_old[pairs_culsum:pairs_culsum+len(pair_idx)] = img_idx
        #     # global_features_mapped[pairs_culsum:pairs_culsum+len(pair_idx)] = global_image_features[img_idx]
        #     pairs_culsum += len(pair_idx)
        # del global_image_features
        # prod_rep = cat(prod_reps, dim=0) # torch.Size([3009, 1536]) torch.Size([5022, 1536]) # # REVIEW: Is this some sort of stateful bug?
        # pair_pred = cat(pair_preds, dim=0) # torch.Size([3009, 2])
        # global_features_mapped[new_to_old] = global_image_features
        #
        #
        # ctx_gate = self.post_cat(prod_rep) # torch.Size([3009, 4096])
        #
        # # use union box and mask convolution
        # if self.use_vision: # True
        #     if self.union_single_not_match: # False
        #         visual_rep = ctx_gate * self.up_dim(union_features)
        #     else:
        #         visual_rep = ctx_gate * union_features # torch.Size([3009, 4096])
        #
        # # GSC Context
        #
        # # 1. Just the context suboutput
        # # 1A. context
        # num_images = [1 for _ in range(global_features_mapped.size(0))]
        # gsc_ctx = self.context_gsc(global_features_mapped, num_images) # torch.Size([506, 4096]) =>
        #
        # # 2. Post context for the overall model
        # gsc_rep = self.post_emb_gsc(gsc_ctx) # [num_unions, 768] => [num_unions, 768]
        #
        #
        # # rel_dists_general = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep) # A: bottlenecked edge * union features; => rels B: bottlenecked edge. torch.Size([3009, 51]) for all 3
        # if not self.with_cleanclf:
        #     rel_dists_general = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep) + self.gsc_compress(gsc_rep) # TODO: this is new # TODO need to match which of the 3009 belong to which image. Need to up dim but with unravling.
        #     if self.use_bias: # True
        #         freq_dists_bias = self.freq_bias.index_with_labels(pair_pred)
        #         freq_dists_bias = F_dropout(freq_dists_bias, 0.3, training=self.training)
        #         rel_dists_general = rel_dists_general + freq_dists_bias # torch.Size([3009, 51])
        #     rel_dists = rel_dists_general
        # # the transfer classifier
        # if self.with_cleanclf:
        #     rel_dists_clean = self.rel_compress_clean(visual_rep) + self.ctx_compress_clean(prod_rep) + self.gsc_compress_clean(global_features_mapped)
        #     if self.use_bias:
        #         freq_dists_bias_clean = self.freq_bias_clean.index_with_labels(pair_pred)
        #         freq_dists_bias_clean = F_dropout(freq_dists_bias_clean, 0.3, training=self.training)
        #         rel_dists_clean = rel_dists_clean + freq_dists_bias_clean
        #     rel_dists = rel_dists_clean
        #
        # if self.with_transfer:
        #     rel_dists = (self.pred_adj_nor @ rel_dists.T).T
        #
        # add_losses = {}
        # # if self.with_knowdist:
        # #     rel_dists_specific_soft = F.log_softmax(rel_dists, -1)
        # #     rel_dists_general_soft = F_softmax(rel_dists_general, -1)
        # #     add_losses['know_dist_kl'] = self.kd_alpha * self.kl_loss(rel_dists_specific_soft, rel_dists_general_soft)
        #
        # obj_dists = obj_dists.split(num_objs, dim=0) # torch.Size([1280, 151]) => 16 * torch.Size([80, 151])
        # rel_dists = rel_dists.split(num_rels, dim=0) # torch.Size([5022, 51]) => (Pdb) rel_dists.split(num_rels, dim=0)[0].size() torch.Size([156, 51]), 240, ...
        #
        # if self.attribute_on:
        #     att_dists = att_dists.split(num_objs, dim=0)
        #     return (obj_dists, att_dists), rel_dists, add_losses
        # return obj_dists, rel_dists, add_losses


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
