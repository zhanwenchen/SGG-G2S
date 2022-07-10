from pickle import load as pickle_load
from torch import (
    as_tensor as torch_as_tensor,
    float32 as torch_float32,
    zeros as torch_zeros,
    cat as torch_cat,
    mm as torch_mm,
    sigmoid as torch_sigmoid,
    tanh as torch_tanh,
    transpose as torch_transpose,
    matmul as torch_matmul,
    addcmul as torch_addcmul,
    arange as torch_arange,
    device as torch_device,
    float16 as torch_float16,
)
from torch.nn import Module, Linear
from torch.nn.functional import softmax as F_softmax
from torch.jit import script as torch_jit_script
from maskrcnn_benchmark.data import get_dataset_statistics
from .utils_relation import MLP, layer_init_kaiming_normal
from .utils_motifs import to_onehot


class GGNNContext(Module):
    def __init__(self, config):
        super().__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters`
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.dropout_rate = config.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE

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
        if config.MODEL.ROI_RELATION_HEAD.GBNET.USE_EMBEDDING is True:
            with open(config.MODEL.ROI_RELATION_HEAD.GBNET.EMB_PATH, 'rb') as fin:
                self.emb_ent, self.emb_pred = pickle_load(fin)

        if config.MODEL.ROI_RELATION_HEAD.GBNET.USE_KNOWLEDGE is True:
            with open(config.MODEL.ROI_RELATION_HEAD.GBNET.KB_PATH, 'rb') as fin:
                edge_dict = pickle_load(fin)

            self.adjmtx_ent2ent = edge_dict['edges_ent2ent']
            self.adjmtx_ent2pred = edge_dict['edges_ent2pred']
            self.adjmtx_pred2ent = edge_dict['edges_pred2ent']
            self.adjmtx_pred2pred = edge_dict['edges_pred2pred']

        self.time_step_num = config.MODEL.ROI_RELATION_HEAD.GBNET.TIME_STEP_NUM
        self.num_edge_types_ent2ent = self.adjmtx_ent2ent.shape[0]
        self.num_edge_types_ent2pred = self.adjmtx_ent2pred.shape[0]
        self.num_edge_types_pred2ent = self.adjmtx_pred2ent.shape[0]
        self.num_edge_types_pred2pred = self.adjmtx_pred2pred.shape[0]

        hidden_dim = config.MODEL.ROI_RELATION_HEAD.GBNET.HIDDEN_DIM
        self.fc_init_ont_ent = Linear(self.emb_ent.shape[1], hidden_dim) # 300, 1024
        # layer_init_kaiming_normal(self.fc_init_ont_ent)
        self.fc_init_ont_pred = Linear(self.emb_pred.shape[1], hidden_dim)
        # layer_init_kaiming_normal(self.fc_init_ont_pred)

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
        # layer_init_kaiming_normal(self.fc_eq3_w_ont_ent)
        self.fc_eq3_u_ont_ent = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq3_u_ont_ent)
        self.fc_eq4_w_ont_ent = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq4_w_ont_ent)
        self.fc_eq4_u_ont_ent = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq4_u_ont_ent)
        self.fc_eq5_w_ont_ent = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq5_w_ont_ent)
        self.fc_eq5_u_ont_ent = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq5_u_ont_ent)

        self.fc_eq3_w_ont_pred = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq3_w_ont_pred)
        self.fc_eq3_u_ont_pred = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq3_u_ont_pred)
        self.fc_eq4_w_ont_pred = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq4_w_ont_pred)
        self.fc_eq4_u_ont_pred = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq4_u_ont_pred)
        self.fc_eq5_w_ont_pred = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq5_w_ont_pred)
        self.fc_eq5_u_ont_pred = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq5_u_ont_pred)

        self.fc_eq3_w_img_ent = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq3_w_img_ent)
        self.fc_eq3_u_img_ent = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq3_u_img_ent)
        self.fc_eq4_w_img_ent = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq4_w_img_ent)
        self.fc_eq4_u_img_ent = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq4_u_img_ent)
        self.fc_eq5_w_img_ent = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq5_w_img_ent)
        self.fc_eq5_u_img_ent = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq5_u_img_ent)

        self.fc_eq3_w_img_pred = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq3_w_img_pred)
        self.fc_eq3_u_img_pred = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq3_u_img_pred)
        self.fc_eq4_w_img_pred = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq4_w_img_pred)
        self.fc_eq4_u_img_pred = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq4_u_img_pred)
        self.fc_eq5_w_img_pred = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq5_w_img_pred)
        self.fc_eq5_u_img_pred = Linear(hidden_dim, hidden_dim, device=device)
        # layer_init_kaiming_normal(self.fc_eq5_u_img_pred)

        self.fc_output_proj_img_pred = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False, device=device)
        self.fc_output_proj_ont_pred = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False, device=device)

        # TODO: isn't refine_obj_cls determined by mode? Let's just do config now
        self.refine_obj_cls = config.MODEL.ROI_RELATION_HEAD.GBNET.REFINE_OBJ_CLS
        if self.refine_obj_cls:
            self.fc_output_proj_img_ent = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False, device=device)
            self.fc_output_proj_ont_ent = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False, device=device)


    def forward(self, proposals, rel_pair_idxs, roi_features_images, union_features_images, device=None, debug=True):
        device = union_features_images.device
        dtype = union_features_images.dtype
        obj_labels = torch_cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        obj_logits = to_onehot(obj_labels, self.num_obj_cls).type(dtype)
        obj_probs = F_softmax(obj_logits, dtype=torch_float32, dim=1).type(dtype) #TODO: type_as?

        num_objs = [len(proposal) for proposal in proposals]
        num_objs_sum_batch = sum(num_objs)
        num_rels_sum_batch = sum([len(rel_pair_idx) for rel_pair_idx in rel_pair_idxs])
        # num_objs = len(proposal) #0: 15
        # num_rels = len(rel_pair_idx) #0: 210
        # num_obj_cls = self.num_obj_cls
        num_rel_cls = self.num_rel_cls
        nodes_ont_ent = self.fc_init_ont_ent(torch_as_tensor(self.emb_ent, dtype=dtype, device=device))
        nodes_ont_pred = self.fc_init_ont_pred(torch_as_tensor(self.emb_pred, dtype=dtype, device=device))
        # nodes_img_ent = roi_features[idx_start_ent:idx_start_ent+num_objs]
        nodes_img_ent = roi_features_images
        nodes_img_pred = union_features_images

        # TODO: use placeholders instead of generating new tensors which
        # require a recreation of the computation graph. This may be true
        # even if requires_grad = False.
        edges_ont_ent2ent = torch_as_tensor(self.adjmtx_ent2ent, dtype=dtype, device=device)
        edges_ont_ent2pred = torch_as_tensor(self.adjmtx_ent2pred, dtype=dtype, device=device)
        edges_ont_pred2ent = torch_as_tensor(self.adjmtx_pred2ent, dtype=dtype, device=device)
        edges_ont_pred2pred = torch_as_tensor(self.adjmtx_pred2pred, dtype=dtype, device=device)

        obj_offset = 0
        rel_offset = 0
        sub_global_inds = []
        obj_global_inds = []
        # TODO: is this where the bug is - that I need to index the rel results?
        # rel_global_inds = []
        for pair_idx, num_obj in zip(rel_pair_idxs, num_objs):
            num_rel = pair_idx.shape[0]
            sub_idx = pair_idx[:,0].contiguous().long().view(-1) + obj_offset
            obj_idx = pair_idx[:,1].contiguous().long().view(-1) + obj_offset
            # rel_idx = torch_arange(num_rel, device=device, dtype=torch_int64).view(-1) + rel_offset

            sub_global_inds.append(sub_idx)
            obj_global_inds.append(obj_idx)

            # sub2rel[sub_idx, rel_idx] = 1.0
            # obj2rel[obj_idx, rel_idx] = 1.0

            obj_offset += num_obj
            rel_offset += num_rel

        sub_global_inds = torch_cat(sub_global_inds, dim=0)
        obj_global_inds = torch_cat(obj_global_inds, dim=0)
        # rel_global_inds = torch_cat(rel_global_inds, dim=0)

        edges_img_pred2subj = torch_zeros((num_rels_sum_batch, num_objs_sum_batch), dtype=dtype, device=device)
        # TODO: # DEBUG: arange vs :? Let's compare
        edges_img_pred2subj[:, sub_global_inds] = 1
        edges_img_pred2obj = torch_zeros((num_rels_sum_batch, num_objs_sum_batch), dtype=dtype, device=device)
        edges_img_pred2obj[:, obj_global_inds] = 1
        edges_img_subj2pred = edges_img_pred2subj.t() # TODO: need to specify axes when vectorized
        edges_img_obj2pred = edges_img_pred2obj.t()

        edges_img2ont_ent = obj_probs.detach().clone()
        edges_ont2img_ent = edges_img2ont_ent.t()

        edges_img2ont_pred = torch_zeros((num_rels_sum_batch, num_rel_cls), dtype=dtype, device=device, requires_grad=False) # torch.Size([506, 51])
        edges_ont2img_pred = edges_img2ont_pred.t()

        ent_cls_logits = None

        num_edge_types_ent2ent = self.num_edge_types_ent2ent
        num_edge_types_pred2ent = self.num_edge_types_pred2ent
        num_edge_types_ent2pred = self.num_edge_types_ent2pred
        num_edge_types_pred2pred = self.num_edge_types_pred2pred

        for t in range(self.time_step_num):
            message_send_ont_ent = self.fc_mp_send_ont_ent(nodes_ont_ent) # torch.Size([177, 1024]) => torch.Size([177, 256])
            message_send_ont_pred = self.fc_mp_send_ont_pred(nodes_ont_pred) # torch.Size([51, 1024]) => torch.Size([51, 256])
            message_send_img_ent = self.fc_mp_send_img_ent(nodes_img_ent) # torch.Size([23, 1024]) => torch.Size([23, 256]) # Can be vectorized
            message_send_img_pred = self.fc_mp_send_img_pred(nodes_img_pred) # torch.Size([506, 1024]) => torch.Size([506, 256]) # Can be vectorized

            # breakpoint(). edges_img2ont_ent.size(). message_send_img_ent.size()
            message_received_ont_ent = torch_cat(
                [torch_mm(edges_ont_ent2ent[i].t(), message_send_ont_ent) for i in range(num_edge_types_ent2ent)] + # 177, 177 => 177, 256 x 9
                [torch_mm(edges_ont_pred2ent[i].t(), message_send_ont_pred) for i in range(num_edge_types_pred2ent)] + # 51 177 => 51, 256 * 3
                [torch_mm(edges_img2ont_ent.t(), message_send_img_ent),] # 1 x [151, 256]
            , 1) #  torch.Size([177, 3328])
            message_received_ont_ent = self.fc_mp_receive_ont_ent(message_received_ont_ent)
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
            ], 1) # torch.Size([with_clean_classifier(506, 768])
            message_received_img_pred = self.fc_mp_receive_img_pred(message_received_img_pred) # torch.Size([506, 768])=>torch.Size([506, 1024])

            del message_send_ont_pred, message_send_img_ent

            nodes_ont_ent = apply_formula_old(message_received_ont_ent, nodes_ont_ent, self.fc_eq3_w_ont_ent, self.fc_eq3_u_ont_ent, self.fc_eq4_w_ont_ent, self.fc_eq4_u_ont_ent, self.fc_eq5_w_ont_ent, self.fc_eq5_u_ont_ent, num_rels_sum_batch)
            nodes_ont_pred = apply_formula_old(message_received_ont_pred, nodes_ont_pred, self.fc_eq3_w_ont_pred, self.fc_eq3_u_ont_pred, self.fc_eq4_w_ont_pred, self.fc_eq4_u_ont_pred, self.fc_eq5_w_ont_pred, self.fc_eq5_u_ont_pred, num_rels_sum_batch)
            nodes_img_ent = apply_formula_old(message_received_img_ent, nodes_img_ent, self.fc_eq3_w_img_ent, self.fc_eq3_u_img_ent, self.fc_eq4_w_img_ent, self.fc_eq4_u_img_ent, self.fc_eq5_w_img_ent, self.fc_eq5_u_img_ent, num_rels_sum_batch)
            nodes_img_pred = apply_formula_old(message_received_img_pred, nodes_img_pred, self.fc_eq3_w_img_pred, self.fc_eq3_u_img_pred, self.fc_eq4_w_img_pred, self.fc_eq4_u_img_pred, self.fc_eq5_w_img_pred, self.fc_eq5_u_img_pred, num_rels_sum_batch)

            pred_cls_logits = torch_mm(self.fc_output_proj_img_pred(nodes_img_pred), self.fc_output_proj_ont_pred(nodes_ont_pred).t()) # torch.Size([506, 1024]) @ torch.Size([1024, 51])

            edges_img2ont_pred = F_softmax(pred_cls_logits, dtype=torch_float32, dim=1).type_as(pred_cls_logits)
            edges_ont2img_pred = edges_img2ont_pred.t()
            if self.refine_obj_cls:
                ent_cls_logits = torch_mm(self.fc_output_proj_img_ent(nodes_img_ent), self.fc_output_proj_ont_ent(nodes_ont_ent).t())
                edges_img2ont_ent = F_softmax(ent_cls_logits, dtype=torch_float32, dim=1).type_as(ent_cls_logits)
                edges_ont2img_ent = edges_img2ont_ent.t()
            else:
                ent_cls_logits = obj_logits

        return ent_cls_logits, pred_cls_logits


def original(fc, a1, b1, a2, b2, a3, b3):
    return fc(torch_cat(
    [torch_mm(a1[i].t(), b1) for i in range(a1.size(0))] + # 177, 177 => 177, 256 x 9
    [torch_mm(a2[i].t(), b2) for i in range(a2.size(0))] + # 51 177 => 51, 256 * 3
    [torch_mm(a3.t(), b3),]
    , 1))

def jit_transpose_first_inplace(fc, a1, b1, a2, b2, a3, b3):
    return fc(mm_flatten_cat_transpose_first_inplace(a1, b1, a2, b2, a3, b3))
#
def nonjit_transpose_first(fc, a1, b1, a2, b2, a3, b3):
    return fc(mm_flatten_cat_transpose_first_nonjit(a1, b1, a2, b2, a3, b3))

def get_new_nodes_img_pred_old(self, message_received_img_pred, nodes_img_pred):
    z_img_pred = torch_sigmoid(self.fc_eq3_w_img_pred(message_received_img_pred) + self.fc_eq3_u_img_pred(nodes_img_pred)) # torch.Size([506, 1024])
    r_img_pred = torch_sigmoid(self.fc_eq4_w_img_pred(message_received_img_pred) + self.fc_eq4_u_img_pred(nodes_img_pred))
    h_img_pred = torch_tanh(self.fc_eq5_w_img_pred(message_received_img_pred) + self.fc_eq5_u_img_pred(r_img_pred * nodes_img_pred))
    # del message_received_img_pred, r_img_pred
    return (1 - z_img_pred) * nodes_img_pred + z_img_pred * h_img_pred # nodes_img_pred: torch.Size([506, 1024])

def apply_formula_old(x, y, eq3x, eq3y, eq4x, eq4y, eq5x, eq5y, num_rels):
    return formula_old(y,
    eq3x(x),
    eq3y(y),
    eq5x(x),
    eq5y(eq4x(x).add_(eq4y(y)).sigmoid_() * y),
    )

def apply_formula(x, y, eq3x, eq3y, eq4x, eq4y, eq5x, eq5y, num_rels):
    if num_rels <= 60:
        return formula_oneline(y,
            eq3x(x),
            eq3y(y),
            eq5x(x),
            eq5y(eq4x(x).add_(eq4y(y)).sigmoid_() * y),
        )
    return formula_addcmul(y,
        eq3x(x),
        eq3y(y),
        eq5x(x),
        eq5y(addsigmoidmuly(eq4x(x), eq4y(y), y)),
    )

def mm_flatten_cat_transpose_first_nonjit(a1, b1, a2, b2, a3, b3):
    return torch_cat((
        torch_transpose(torch_matmul(torch_transpose(a1, 1, 2), b1), 0, 1),
        torch_transpose(torch_matmul(torch_transpose(a2, 1, 2), b2), 0, 1),
        torch_mm(torch_transpose(a3, 0, 1), b3).unsqueeze_(1),
    ), dim=1).view(a3.size(1), -1)

@torch_jit_script
def mm_flatten_cat_transpose_first_inplace(a1, b1, a2, b2, a3, b3):
    return torch_cat((
        torch_matmul(a1.transpose(1, 2), b1).transpose_(0, 1),
        torch_matmul(a2.transpose(1, 2), b2).transpose_(0, 1),
        torch_mm(a3.transpose(0, 1), b3).unsqueeze_(1),
    ), dim=1).view(a3.size(1), -1)

@torch_jit_script
def addsigmoid(a, b):
    return torch_sigmoid(a + b)

@torch_jit_script
def addsigmoidmuly(a, b, y):
    return torch_sigmoid(a + b) * y

# @torch_jit_script
# def addtahn(a, b):
#     return torch_tanh(a + b)

# @torch_jit_script
def formula_old(x, a, b, e, f):
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

# No benefit to addcmul over the regular formula
@torch_jit_script
def formula_addcmul(x, a, b, e, f):
    z = torch_sigmoid(a + b) # torch.Size([23, 1024])
    del a, b
    # r_img_ent = torch_sigmoid(c + d)
    # TODO: try return (z * torch_tanh(e + f).addcmul_, (1 - z), x) # nodes_img_ent: torch.Size([23, 1024])
    return torch_addcmul((1 - z) * x, z, torch_tanh(e + f)) # nodes_img_ent: torch.Size([23, 1024])
