"""
Based on the implementation of https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""
from torch import bmm as torch_bmm, no_grad as torch_no_grad, arange as torch_arange, LongTensor as torch_LongTensor, cat as torch_cat
from torch.nn import Module, ModuleList, Sequential, Dropout, Softmax, Linear, Conv1d, ReLU, LayerNorm, Embedding
from torch.nn.functional import softmax as F_softmax, relu as F_relu
from torch.nn.init import normal_, xavier_normal_
from torch.nn.utils.rnn import pad_sequence
from numpy import inf as np_inf, sqrt as np_sqrt, power as np_power, unravel_index as np_unravel_index
from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info


class ScaledDotProductAttention(Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = Dropout(attn_dropout)
        self.softmax = Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (bsz, len_q, dim_q)
            k (bsz, len_k, dim_k)
            v (bsz, len_v, dim_v)
            Note: len_k==len_v, and dim_q==dim_k
        Returns:
            output (bsz, len_q, dim_v)
            attn (bsz, len_q, len_k)
        """
        attn = torch_bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np_inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch_bmm(attn, v)

        return output, attn


class MultiHeadAttention(Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = Linear(d_model, n_head * d_k)
        self.w_ks = Linear(d_model, n_head * d_k)
        self.w_vs = Linear(d_model, n_head * d_v)
        normal_(self.w_qs.weight, mean=0, std=np_sqrt(2.0 / (d_model + d_k)))
        normal_(self.w_ks.weight, mean=0, std=np_sqrt(2.0 / (d_model + d_k)))
        normal_(self.w_vs.weight, mean=0, std=np_sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np_power(d_k, 0.5))
        self.layer_norm = LayerNorm(d_model)

        self.fc = Linear(n_head * d_v, d_model)
        xavier_normal_(self.fc.weight)

        self.dropout = Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (bsz, len_q, dim_q)
            k (bsz, len_k, dim_k)
            v (bsz, len_v, dim_v)
            Note: len_k==len_v, and dim_q==dim_k
        Returns:
            output (bsz, len_q, d_model)
            attn (bsz, len_q, len_k)
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()  # len_k==len_v

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = LayerNorm(d_in)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        """
        Merge adjacent information. Equal to linear layer if kernel size is 1
        Args:
            x (bsz, len, dim)
        Returns:
            output (bsz, len, dim)
        """
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F_relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask.float()

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask.float()

        return enc_output, enc_slf_attn


class TransformerEncoder(Module):
    """
    A encoder model with self attention mechanism.
    """

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.layer_stack = ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, input_feats, num_objs):
        """
        Args:
            input_feats [Tensor] (#total_box, d_model) : bounding box features of a batch
            num_objs [list of int] (bsz, ) : number of bounding box of each image
        Returns:
            enc_output [Tensor] (#total_box, d_model)
        """
        input_feats = input_feats.split(num_objs, dim=0)
        input_feats = pad_sequence(input_feats, batch_first=True)

        # -- Prepare masks
        bsz = len(num_objs)
        device = input_feats.device
        pad_len = max(num_objs)
        num_objs_ = torch_LongTensor(num_objs).to(device).unsqueeze(1).expand(-1, pad_len)
        slf_attn_mask = torch_arange(pad_len, device=device).view(1, -1).expand(bsz, -1).ge(num_objs_).unsqueeze(
            1).expand(-1, pad_len, -1)  # (bsz, pad_len, pad_len)
        non_pad_mask = torch_arange(pad_len, device=device).to(device).view(1, -1).expand(bsz, -1).lt(
            num_objs_).unsqueeze(-1)  # (bsz, pad_len, 1)

        # -- Forward
        enc_output = input_feats
        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        enc_output = enc_output[non_pad_mask.squeeze(-1)]
        return enc_output


class TransformerContext(Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super().__init__()
        self.cfg = config
        # setting parameters
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER
        self.edge_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # the following word embedding layer should be initalize by glove.6B before using
        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)

        self.obj_embed1 = Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = Embedding(self.num_obj_cls, self.embed_dim)
        with torch_no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)

        # position embedding
        self.bbox_embed = Sequential(*[
            Linear(9, 32), ReLU(inplace=True), Dropout(0.1),
            Linear(32, 128), ReLU(inplace=True), Dropout(0.1),
        ])
        self.lin_obj = Linear(self.in_channels + self.embed_dim + 128, self.hidden_dim)
        self.lin_edge = Linear(self.embed_dim + self.hidden_dim + self.in_channels, self.hidden_dim)
        self.out_obj = Linear(self.hidden_dim, self.num_obj_cls)
        self.context_obj = TransformerEncoder(self.obj_layer, self.num_head, self.k_dim,
                                              self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)
        self.context_edge = TransformerEncoder(self.edge_layer, self.num_head, self.k_dim,
                                               self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)

    def forward(self, roi_features, proposals, logger=None, features=None):
        '''
        Args:
            roi_features:
                (Pdb) roi_features.size()
                torch.Size([1280, 4096]): 80 proposals per image. 80 * 16 = 1280
            proposals:
                (Pdb) num_objs
                [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]
                Each has label and logits.
            features (mine):
                # (Pdb) len(features)
                # 5
                # (Pdb) features[0].size()
                # torch.Size([16, 256, 256, 152])
                # (Pdb) features[1].size()
                # torch.Size([16, 256, 128, 76])
                # (Pdb) features[2].size()
                # torch.Size([16, 256, 64, 38])
                # (Pdb) features[3].size()
                # torch.Size([16, 256, 32, 19])
                # (Pdb) features[4].size()
                # torch.Size([16, 256, 16, 10])
        Returns:
            (Pdb) obj_dists.size()
            torch.Size([1280, 151])

        '''
        # labels will be used in DecoderRNN during training
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        # (Pdb) obj_labels.size()
        # torch.Size([1280])

        if obj_labels is not None:
            obj_labels = obj_labels.long()
        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F_softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        # (Pdb) obj_embed.size()
        # torch.Size([1280, 200])

        # bbox embedding will be used as input
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.bbox_embed(encode_box_info(proposals))
        # (Pdb) pos_embed.size()
        # torch.Size([1280, 128])

        # encode objects with transformer
        obj_pre_rep = cat((roi_features, obj_embed, pos_embed), -1)
        num_objs = [len(p) for p in proposals]
        obj_pre_rep = self.lin_obj(obj_pre_rep)
        # (Pdb) obj_pre_rep.size()
        # torch.Size([1280, 768])
        obj_feats = self.context_obj(obj_pre_rep, num_objs)
        # (Pdb) obj_feats.size()
        # torch.Size([1280, 768])

        # predict obj_dists and obj_preds
        if self.mode == 'predcls':
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
            # (Pdb) obj_dists.size()
            # torch.Size([1280, 151])
            # edge_pre_rep = cat((roi_features, obj_feats, features, self.obj_embed2(obj_labels)), dim=-1)
            edge_pre_rep = cat((roi_features, obj_feats, self.obj_embed2(obj_labels)), dim=-1)
        else:
            obj_dists = self.out_obj(obj_feats)
            # (Pdb) obj_dists.size()
            # torch.Size([1280, 151])
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1
                # obj_preds are object predictions.
                # (Pdb) obj_preds.size()
                # torch.Size([1280])
                # What is obj_preds?

            # edge_pre_rep = cat((roi_features, obj_feats, features, self.obj_embed2(obj_preds)), dim=-1)
            edge_pre_rep = cat((roi_features, obj_feats, self.obj_embed2(obj_preds)), dim=-1)

        # edge context
        # (Pdb) edge_pre_rep.size()
        # torch.Size([1280, 5064])
        edge_pre_rep = self.lin_edge(edge_pre_rep)
        # (Pdb) edge_pre_rep.size()
        # torch.Size([1280, 768])

        edge_ctx = self.context_edge(edge_pre_rep, num_objs)
        # (Pdb) edge_ctx.size()
        # torch.Size([1280, 768])

        return obj_dists, obj_preds, edge_ctx

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)

            out_dists_sampled = F_softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np_unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label.long())
        return torch_cat(obj_preds, dim=0)
