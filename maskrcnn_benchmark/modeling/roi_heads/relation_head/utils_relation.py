from itertools import product as itertools_product
from torch import (
    min as torch_min,
    max as torch_max,
    clamp as torch_clamp,
    cat as torch_cat,
    nonzero as torch_nonzero,
    zeros as torch_zeros,
    int64 as torch_int64,
    prod as torch_prod,
)
from torch.nn import Module, Sequential, Linear, ReLU
from torch.nn.init import kaiming_normal_, constant_, xavier_normal_, normal_, orthogonal_
from torch.nn.functional import softmax as F_softmax
from numpy import unravel_index as np_unravel_index


def get_box_info(boxes, need_norm=True, proposal=None):
    """
    input: [batch_size, (x1,y1,x2,y2)]
    output: [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    """
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0
    center_box = torch_cat((boxes[:, :2] + 0.5 * wh, wh), 1)
    box_info = torch_cat((boxes, center_box), 1)
    del boxes, center_box
    if need_norm:
        box_info = box_info / float(max(max(proposal.size[0], proposal.size[1]), 100))
    return box_info

def get_box_pair_info(box1, box2):
    """
    input:
        box1 [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
        box2 [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    output:
        32-digits: [box1, box2, unionbox, intersectionbox]
    """
    # union box
    unionbox = box1[:,:4].clone()
    unionbox[:, 0] = torch_min(box1[:, 0], box2[:, 0])
    unionbox[:, 1] = torch_min(box1[:, 1], box2[:, 1])
    unionbox[:, 2] = torch_max(box1[:, 2], box2[:, 2])
    unionbox[:, 3] = torch_max(box1[:, 3], box2[:, 3])
    union_info = get_box_info(unionbox, need_norm=False)
    del unionbox

    # intersection box
    intersextion_box = box1[:,:4].clone()
    intersextion_box[:, 0] = torch_max(box1[:, 0], box2[:, 0])
    intersextion_box[:, 1] = torch_max(box1[:, 1], box2[:, 1])
    intersextion_box[:, 2] = torch_min(box1[:, 2], box2[:, 2])
    intersextion_box[:, 3] = torch_min(box1[:, 3], box2[:, 3])
    case1 = torch_nonzero(intersextion_box[:, 2].contiguous().view(-1) < intersextion_box[:, 0].contiguous().view(-1)).view(-1)
    case2 = torch_nonzero(intersextion_box[:, 3].contiguous().view(-1) < intersextion_box[:, 1].contiguous().view(-1)).view(-1)
    intersextion_info = get_box_info(intersextion_box, need_norm=False)
    del intersextion_box
    if case1.numel() > 0:
        intersextion_info[case1, :] = 0
    if case2.numel() > 0:
        intersextion_info[case2, :] = 0
    return torch_cat((box1, box2, union_info, intersextion_info), 1)

def nms_overlaps(boxes):
    """ get overlaps for each channel"""
    assert boxes.dim() == 3
    # breakpoint()
    # N = boxes.size(0)
    # nc = boxes.size(1)
    N, nc, _ = boxes.size()
    max_xy = torch_min(boxes[:, None, :, 2:].expand(N, N, nc, 2),
                       boxes[None, :, :, 2:].expand(N, N, nc, 2))

    min_xy = torch_max(boxes[:, None, :, :2].expand(N, N, nc, 2),
                       boxes[None, :, :, :2].expand(N, N, nc, 2))

    inter = torch_clamp((max_xy - min_xy + 1.0), min=0)
    del max_xy, min_xy
    # n, n, 151
    # inters = inter[:,:,:,0]*inter[:,:,:,1]
    inters = torch_prod(inter, 3)
    del inter
    boxes_flat = boxes.view(-1, 4)
    del boxes
    areas_flat = (boxes_flat[:,2]- boxes_flat[:,0]+1.0)*(
        boxes_flat[:,3]- boxes_flat[:,1]+1.0)
    # areas = areas_flat.view(boxes.size(0), boxes.size(1))
    areas = areas_flat.view(N, nc)
    union = -inters + areas[None] + areas[:, None]
    return inters / union


# def layer_init(layer, init_para=0.1, normal=False, xavier=True):
#     xavier = False if normal == True else True
#     if normal:
#         normal_(layer.weight, mean=0, std=init_para)
#         if layer.bias is not None:
#             constant_(layer.bias, 0)
#         return
#     elif xavier:
#         xavier_normal_(layer.weight, gain=1.0)
#         if layer.bias is not None:
#             constant_(layer.bias, 0)
#         return


def layer_init_kaiming_normal(layer):
    kaiming_normal_(layer.weight, mode='fan_out')
    if layer.bias is not None:
        constant_(layer.bias, 0)


class XavierLinear(Module):
    '''
    Simple Linear layer with Xavier init

    Paper by Xavier Glorot and Yoshua Bengio (2010):
    Understanding the difficulty of training deep feedforward neural networks
    http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    '''
    def __init__(self, in_features, out_features, bias=True, device=None):
        if device is None:
            raise
        super(XavierLinear, self).__init__()
        self.linear = Linear(in_features, out_features, bias=bias, device=device)
        layer_init_kaiming_normal(self.linear)

    def forward(self, x):
        return self.linear(x)


class MLP(Module):
    def __init__(self, dim_in_hid_out, act_fn='ReLU', last_act=False, device=None):
        if device is None:
            raise
        super(MLP, self).__init__()
        layers = []
        for i in range(len(dim_in_hid_out) - 1):
            layers.append(XavierLinear(dim_in_hid_out[i], dim_in_hid_out[i + 1], device=device))
            if i < len(dim_in_hid_out) - 2 or last_act:
                if act_fn == 'ReLU':
                    layers.append(ReLU())
                else:
                    raise
        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def obj_prediction_nms(boxes_per_cls, pred_logits, nms_thresh=0.3):
    """
    boxes_per_cls:               [num_obj, num_cls, 4]
    pred_logits:                 [num_obj, num_category]
    """
    num_obj = pred_logits.shape[0]
    assert num_obj == boxes_per_cls.shape[0]

    is_overlap = nms_overlaps(boxes_per_cls).view(boxes_per_cls.size(0), boxes_per_cls.size(0),
                              boxes_per_cls.size(1)).cpu().numpy() >= nms_thresh

    prob_sampled = F_softmax(pred_logits, 1).cpu().numpy()
    prob_sampled[:, 0] = 0  # set bg to 0

    pred_label = torch_zeros(num_obj, device=pred_logits.device, dtype=torch_int64)

    for i in range(num_obj):
        box_ind, cls_ind = np_unravel_index(prob_sampled.argmax(), prob_sampled.shape)
        if float(pred_label[int(box_ind)]) > 0:
            pass
        else:
            pred_label[int(box_ind)] = int(cls_ind)
        prob_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
        prob_sampled[box_ind] = -1.0 # This way we won't re-sample

    return pred_label


def block_orthogonal(tensor, split_sizes, gain=1.0):
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ValueError("tensor dimensions must be divisible by their respective "
                         "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split))
               for max_size, split in zip(sizes, split_sizes)]
    # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools_product(*indexes):
        # A list of tuples containing the index to start at for this block
        # and the appropriate step size (i.e split_size[i] for dimension i).
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        # This is a tuple of slices corresponding to:
        # tensor[index: index + step_size, ...]. This is
        # required because we could have an arbitrary number
        # of dimensions. The actual slices we need are the
        # start_index: start_index + step for each dimension in the tensor.
        block_slice = tuple([slice(start_index, start_index + step)
                             for start_index, step in index_and_step_tuples])

        # let's not initialize empty things to 0s because THAT SOUNDS REALLY BAD
        assert len(block_slice) == 2
        sizes = [x.stop - x.start for x in block_slice]
        tensor_copy = tensor.new(max(sizes), max(sizes))
        orthogonal_(tensor_copy, gain=gain)
        tensor[block_slice] = tensor_copy[0:sizes[0], 0:sizes[1]]
