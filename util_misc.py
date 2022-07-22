from numpy import prod as np_prod
from torch import load as torch_load, no_grad as torch_no_grad


def print_para(model):
    """
    Prints parameters of a model
    :param opt:
    :return:
    """
    st = {}
    strings = []
    total_params = 0
    for p_name, p in model.named_parameters():
        # if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):
        # if 'roi_heads.relation.predictor' in p_name:
        st[p_name] = ([str(x) for x in p.size()], np_prod(p.size()), p.requires_grad)
        total_params += np_prod(p.size())
    for p_name, (size, prod, p_req_grad) in sorted(st.items(), key=lambda x: -x[1][1]):
        # strings.append(p_name)
        strings.append("{:<50s}: {:<16s}({:8d}) ({})".format(
            p_name, '[{}]'.format(','.join(size)), prod, 'grad' if p_req_grad else '    '
        ))
    return '\n {:.1f}M total parameters \n ----- \n \n{}'.format(total_params / 1000000.0, '\n'.join(strings))


def load_gbnet_fcs_weights(model, fpath, state_dict=None):
    pass


def load_gbnet_rpn_weights(model, fpath, state_dict=None):
    pass


def load_gbnet_vgg_weights(model, fpath, state_dict=None):
    if state_dict is None:
        state_dict = torch_load(fpath)['state_dict']
    with torch_no_grad():
        model.backbone.body.conv_body[0].weight.copy_(state_dict['features.0.weight'])
        model.backbone.body.conv_body[0].bias.copy_(state_dict['features.0.bias'])
        model.backbone.body.conv_body[2].weight.copy_(state_dict['features.2.weight'])
        model.backbone.body.conv_body[2].bias.copy_(state_dict['features.2.bias'])
        model.backbone.body.conv_body[5].weight.copy_(state_dict['features.5.weight'])
        model.backbone.body.conv_body[5].bias.copy_(state_dict['features.5.bias'])
        model.backbone.body.conv_body[7].weight.copy_(state_dict['features.7.weight'])
        model.backbone.body.conv_body[7].bias.copy_(state_dict['features.7.bias'])
        model.backbone.body.conv_body[10].weight.copy_(state_dict['features.10.weight'])
        model.backbone.body.conv_body[10].bias.copy_(state_dict['features.10.bias'])
        model.backbone.body.conv_body[12].weight.copy_(state_dict['features.12.weight'])
        model.backbone.body.conv_body[12].bias.copy_(state_dict['features.12.bias'])
        model.backbone.body.conv_body[14].weight.copy_(state_dict['features.14.weight'])
        model.backbone.body.conv_body[14].bias.copy_(state_dict['features.14.bias'])
        model.backbone.body.conv_body[17].weight.copy_(state_dict['features.17.weight'])
        model.backbone.body.conv_body[17].bias.copy_(state_dict['features.17.bias'])
        model.backbone.body.conv_body[19].weight.copy_(state_dict['features.19.weight'])
        model.backbone.body.conv_body[19].bias.copy_(state_dict['features.19.bias'])
        model.backbone.body.conv_body[21].weight.copy_(state_dict['features.21.weight'])
        model.backbone.body.conv_body[21].bias.copy_(state_dict['features.21.bias'])
        model.backbone.body.conv_body[24].weight.copy_(state_dict['features.24.weight'])
        model.backbone.body.conv_body[24].bias.copy_(state_dict['features.24.bias'])
        model.backbone.body.conv_body[26].weight.copy_(state_dict['features.26.weight'])
        model.backbone.body.conv_body[26].bias.copy_(state_dict['features.26.bias'])
        model.backbone.body.conv_body[28].weight.copy_(state_dict['features.28.weight'])
        model.backbone.body.conv_body[28].bias.copy_(state_dict['features.28.bias'])
    print('loaded model with gbnet vg vgg16 weights')
    return state_dict
 # 'roi_fmap.0.weight',
 # 'roi_fmap.0.bias',
 # 'roi_fmap.3.weight',
 # 'roi_fmap.3.bias',
 # 'score_fc.weight',
 # 'score_fc.bias',
 # 'bbox_fc.weight',
 # 'bbox_fc.bias',
 # 'rpn_head.anchors',
 # 'rpn_head.conv.0.weight',
 # 'rpn_head.conv.0.bias',
 # 'rpn_head.conv.2.weight',
 # 'rpn_head.conv.2.bias']

def load_gbnet_relation(model, fpath):
    state_dict = torch_load(fpath)['state_dict']
    with torch_no_grad():
        model.roi_heads.relation.predictor.ggnn.fc_mp_receive_ont_ent[0].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_ont_ent.model.0.linear.weight'])
        model.roi_heads.relation.predictor.obj_proj.weight.copy_(state_dict['ggnn_rel_reason.obj_proj.weight'])
        model.roi_heads.relation.predictor.rel_proj.weight.copy_(state_dict['ggnn_rel_reason.rel_proj.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_receive_ont_pred[0].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_ont_pred.model.0.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_receive_ont_ent[2].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_ont_ent.model.2.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_receive_ont_pred[2].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_ont_pred.model.2.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq3_w_ont_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_w_ont_ent.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq3_u_ont_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_u_ont_ent.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq4_w_ont_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_w_ont_ent.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq4_u_ont_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_u_ont_ent.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq5_w_ont_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_w_ont_ent.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq5_u_ont_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_u_ont_ent.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq3_w_ont_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_w_ont_pred.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq3_u_ont_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_u_ont_pred.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq4_w_ont_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_w_ont_pred.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq4_u_ont_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_u_ont_pred.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq5_w_ont_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_w_ont_pred.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq5_u_ont_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_u_ont_pred.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq3_w_img_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_w_img_ent.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq3_u_img_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_u_img_ent.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq4_w_img_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_w_img_ent.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq4_u_img_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_u_img_ent.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq5_w_img_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_w_img_ent.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq5_u_img_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_u_img_ent.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq3_w_img_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_w_img_pred.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq3_u_img_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_u_img_pred.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq4_w_img_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_w_img_pred.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq4_u_img_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_u_img_pred.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq5_w_img_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_w_img_pred.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_eq5_u_img_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_u_img_pred.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_output_proj_img_pred[0].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_output_proj_img_pred.model.0.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_output_proj_img_pred[2].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_output_proj_img_pred.model.2.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_output_proj_ont_pred[0].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_output_proj_ont_pred.model.0.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_output_proj_ont_pred[2].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_output_proj_ont_pred.model.2.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_receive_img_ent[2].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_img_ent.model.2.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_receive_img_pred[2].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_img_pred.model.2.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_receive_img_ent[0].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_img_ent.model.0.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_receive_img_pred[0].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_img_pred.model.0.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_send_ont_ent[0].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_ont_ent.model.0.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_send_ont_pred[0].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_ont_pred.model.0.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_send_img_ent[0].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_img_ent.model.0.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_send_img_pred[0].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_img_pred.model.0.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_init_ont_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_init_ont_ent.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_init_ont_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_init_ont_pred.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_send_ont_ent[2].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_ont_ent.model.2.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_send_ont_pred[2].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_ont_pred.model.2.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_send_img_ent[2].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_img_ent.model.2.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_send_img_pred[2].weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_img_pred.model.2.linear.weight'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_receive_ont_ent[0].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_ont_ent.model.0.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_receive_ont_pred[0].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_ont_pred.model.0.linear.bias'])
        model.roi_heads.relation.predictor.obj_proj.bias.copy_(state_dict['ggnn_rel_reason.obj_proj.bias'])
        model.roi_heads.relation.predictor.rel_proj.bias.copy_(state_dict['ggnn_rel_reason.rel_proj.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_init_ont_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_init_ont_ent.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_init_ont_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_init_ont_pred.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_receive_ont_ent[2].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_ont_ent.model.2.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_receive_ont_pred[2].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_ont_pred.model.2.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_receive_img_ent[2].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_img_ent.model.2.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_receive_img_pred[2].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_img_pred.model.2.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq3_w_ont_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_w_ont_ent.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq3_u_ont_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_u_ont_ent.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq4_w_ont_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_w_ont_ent.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq4_u_ont_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_u_ont_ent.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq5_w_ont_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_w_ont_ent.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq5_u_ont_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_u_ont_ent.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq3_w_ont_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_w_ont_pred.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq3_u_ont_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_u_ont_pred.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq4_w_ont_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_w_ont_pred.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq4_u_ont_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_u_ont_pred.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq5_w_ont_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_w_ont_pred.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq5_u_ont_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_u_ont_pred.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq3_w_img_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_w_img_ent.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq3_u_img_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_u_img_ent.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq4_w_img_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_w_img_ent.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq4_u_img_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_u_img_ent.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq5_w_img_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_w_img_ent.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq5_u_img_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_u_img_ent.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq3_w_img_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_w_img_pred.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq3_u_img_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_u_img_pred.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq4_w_img_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_w_img_pred.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq4_u_img_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_u_img_pred.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq5_w_img_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_w_img_pred.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_eq5_u_img_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_u_img_pred.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_output_proj_img_pred[0].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_output_proj_img_pred.model.0.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_output_proj_img_pred[2].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_output_proj_img_pred.model.2.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_output_proj_ont_pred[0].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_output_proj_ont_pred.model.0.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_output_proj_ont_pred[2].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_output_proj_ont_pred.model.2.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_receive_img_ent[0].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_img_ent.model.0.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_receive_img_pred[0].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_img_pred.model.0.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_send_ont_ent[0].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_ont_ent.model.0.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_send_ont_pred[0].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_ont_pred.model.0.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_send_img_ent[0].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_img_ent.model.0.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_send_img_pred[0].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_img_pred.model.0.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_send_ont_ent[2].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_ont_ent.model.2.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_send_ont_pred[2].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_ont_pred.model.2.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_send_img_ent[2].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_img_ent.model.2.linear.bias'])
        model.roi_heads.relation.predictor.ggnn.fc_mp_send_img_pred[2].bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_img_pred.model.2.linear.bias'])
    del state_dict
