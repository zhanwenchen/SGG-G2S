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


def load_gbnet_checkpoint(model, fpath):
    state_dict = torch_load(fpath)['state_dict']
    with torch_no_grad():
        model.roi_heads.relation.predictor.fc_mp_receive_ont_ent.model[0].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_ont_ent.model.0.linear.weight'])
        model.roi_heads.relation.predictor.obj_proj.weight.copy_(state_dict['ggnn_rel_reason.obj_proj.weight'])
        model.roi_heads.relation.predictor.rel_proj.weight.copy_(state_dict['ggnn_rel_reason.rel_proj.weight'])
        model.roi_heads.relation.predictor.fc_mp_receive_ont_pred.model[0].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_ont_pred.model.0.linear.weight'])
        model.roi_heads.relation.predictor.fc_mp_receive_ont_ent.model[2].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_ont_ent.model.2.linear.weight'])
        model.roi_heads.relation.predictor.fc_mp_receive_ont_pred.model[2].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_ont_pred.model.2.linear.weight'])
        model.roi_heads.relation.predictor.fc_eq3_w_ont_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_w_ont_ent.weight'])
        model.roi_heads.relation.predictor.fc_eq3_u_ont_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_u_ont_ent.weight'])
        model.roi_heads.relation.predictor.fc_eq4_w_ont_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_w_ont_ent.weight'])
        model.roi_heads.relation.predictor.fc_eq4_u_ont_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_u_ont_ent.weight'])
        model.roi_heads.relation.predictor.fc_eq5_w_ont_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_w_ont_ent.weight'])
        model.roi_heads.relation.predictor.fc_eq5_u_ont_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_u_ont_ent.weight'])
        model.roi_heads.relation.predictor.fc_eq3_w_ont_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_w_ont_pred.weight'])
        model.roi_heads.relation.predictor.fc_eq3_u_ont_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_u_ont_pred.weight'])
        model.roi_heads.relation.predictor.fc_eq4_w_ont_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_w_ont_pred.weight'])
        model.roi_heads.relation.predictor.fc_eq4_u_ont_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_u_ont_pred.weight'])
        model.roi_heads.relation.predictor.fc_eq5_w_ont_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_w_ont_pred.weight'])
        model.roi_heads.relation.predictor.fc_eq5_u_ont_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_u_ont_pred.weight'])
        model.roi_heads.relation.predictor.fc_eq3_w_img_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_w_img_ent.weight'])
        model.roi_heads.relation.predictor.fc_eq3_u_img_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_u_img_ent.weight'])
        model.roi_heads.relation.predictor.fc_eq4_w_img_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_w_img_ent.weight'])
        model.roi_heads.relation.predictor.fc_eq4_u_img_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_u_img_ent.weight'])
        model.roi_heads.relation.predictor.fc_eq5_w_img_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_w_img_ent.weight'])
        model.roi_heads.relation.predictor.fc_eq5_u_img_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_u_img_ent.weight'])
        model.roi_heads.relation.predictor.fc_eq3_w_img_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_w_img_pred.weight'])
        model.roi_heads.relation.predictor.fc_eq3_u_img_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_u_img_pred.weight'])
        model.roi_heads.relation.predictor.fc_eq4_w_img_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_w_img_pred.weight'])
        model.roi_heads.relation.predictor.fc_eq4_u_img_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_u_img_pred.weight'])
        model.roi_heads.relation.predictor.fc_eq5_w_img_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_w_img_pred.weight'])
        model.roi_heads.relation.predictor.fc_eq5_u_img_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_u_img_pred.weight'])
        model.roi_heads.relation.predictor.fc_output_proj_img_pred.model[0].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_output_proj_img_pred.model.0.linear.weight'])
        model.roi_heads.relation.predictor.fc_output_proj_img_pred.model[2].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_output_proj_img_pred.model.2.linear.weight'])
        model.roi_heads.relation.predictor.fc_output_proj_ont_pred.model[0].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_output_proj_ont_pred.model.0.linear.weight'])
        model.roi_heads.relation.predictor.fc_output_proj_ont_pred.model[2].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_output_proj_ont_pred.model.2.linear.weight'])
        model.roi_heads.relation.predictor.fc_mp_receive_img_ent.model[2].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_img_ent.model.2.linear.weight'])
        model.roi_heads.relation.predictor.fc_mp_receive_img_pred.model[2].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_img_pred.model.2.linear.weight'])
        model.roi_heads.relation.predictor.fc_mp_receive_img_ent.model[0].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_img_ent.model.0.linear.weight'])
        model.roi_heads.relation.predictor.fc_mp_receive_img_pred.model[0].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_img_pred.model.0.linear.weight'])
        model.roi_heads.relation.predictor.fc_mp_send_ont_ent.model[0].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_ont_ent.model.0.linear.weight'])
        model.roi_heads.relation.predictor.fc_mp_send_ont_pred.model[0].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_ont_pred.model.0.linear.weight'])
        model.roi_heads.relation.predictor.fc_mp_send_img_ent.model[0].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_img_ent.model.0.linear.weight'])
        model.roi_heads.relation.predictor.fc_mp_send_img_pred.model[0].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_img_pred.model.0.linear.weight'])
        model.roi_heads.relation.predictor.fc_init_ont_ent.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_init_ont_ent.weight'])
        model.roi_heads.relation.predictor.fc_init_ont_pred.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_init_ont_pred.weight'])
        model.roi_heads.relation.predictor.fc_mp_send_ont_ent.model[2].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_ont_ent.model.2.linear.weight'])
        model.roi_heads.relation.predictor.fc_mp_send_ont_pred.model[2].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_ont_pred.model.2.linear.weight'])
        model.roi_heads.relation.predictor.fc_mp_send_img_ent.model[2].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_img_ent.model.2.linear.weight'])
        model.roi_heads.relation.predictor.fc_mp_send_img_pred.model[2].linear.weight.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_img_pred.model.2.linear.weight'])
        model.roi_heads.relation.predictor.fc_mp_receive_ont_ent.model[0].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_ont_ent.model.0.linear.bias'])
        model.roi_heads.relation.predictor.fc_mp_receive_ont_pred.model[0].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_ont_pred.model.0.linear.bias'])
        model.roi_heads.relation.predictor.obj_proj.bias.copy_(state_dict['ggnn_rel_reason.obj_proj.bias'])
        model.roi_heads.relation.predictor.rel_proj.bias.copy_(state_dict['ggnn_rel_reason.rel_proj.bias'])
        model.roi_heads.relation.predictor.fc_init_ont_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_init_ont_ent.bias'])
        model.roi_heads.relation.predictor.fc_init_ont_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_init_ont_pred.bias'])
        model.roi_heads.relation.predictor.fc_mp_receive_ont_ent.model[2].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_ont_ent.model.2.linear.bias'])
        model.roi_heads.relation.predictor.fc_mp_receive_ont_pred.model[2].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_ont_pred.model.2.linear.bias'])
        model.roi_heads.relation.predictor.fc_mp_receive_img_ent.model[2].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_img_ent.model.2.linear.bias'])
        model.roi_heads.relation.predictor.fc_mp_receive_img_pred.model[2].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_img_pred.model.2.linear.bias'])
        model.roi_heads.relation.predictor.fc_eq3_w_ont_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_w_ont_ent.bias'])
        model.roi_heads.relation.predictor.fc_eq3_u_ont_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_u_ont_ent.bias'])
        model.roi_heads.relation.predictor.fc_eq4_w_ont_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_w_ont_ent.bias'])
        model.roi_heads.relation.predictor.fc_eq4_u_ont_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_u_ont_ent.bias'])
        model.roi_heads.relation.predictor.fc_eq5_w_ont_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_w_ont_ent.bias'])
        model.roi_heads.relation.predictor.fc_eq5_u_ont_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_u_ont_ent.bias'])
        model.roi_heads.relation.predictor.fc_eq3_w_ont_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_w_ont_pred.bias'])
        model.roi_heads.relation.predictor.fc_eq3_u_ont_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_u_ont_pred.bias'])
        model.roi_heads.relation.predictor.fc_eq4_w_ont_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_w_ont_pred.bias'])
        model.roi_heads.relation.predictor.fc_eq4_u_ont_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_u_ont_pred.bias'])
        model.roi_heads.relation.predictor.fc_eq5_w_ont_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_w_ont_pred.bias'])
        model.roi_heads.relation.predictor.fc_eq5_u_ont_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_u_ont_pred.bias'])
        model.roi_heads.relation.predictor.fc_eq3_w_img_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_w_img_ent.bias'])
        model.roi_heads.relation.predictor.fc_eq3_u_img_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_u_img_ent.bias'])
        model.roi_heads.relation.predictor.fc_eq4_w_img_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_w_img_ent.bias'])
        model.roi_heads.relation.predictor.fc_eq4_u_img_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_u_img_ent.bias'])
        model.roi_heads.relation.predictor.fc_eq5_w_img_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_w_img_ent.bias'])
        model.roi_heads.relation.predictor.fc_eq5_u_img_ent.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_u_img_ent.bias'])
        model.roi_heads.relation.predictor.fc_eq3_w_img_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_w_img_pred.bias'])
        model.roi_heads.relation.predictor.fc_eq3_u_img_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq3_u_img_pred.bias'])
        model.roi_heads.relation.predictor.fc_eq4_w_img_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_w_img_pred.bias'])
        model.roi_heads.relation.predictor.fc_eq4_u_img_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq4_u_img_pred.bias'])
        model.roi_heads.relation.predictor.fc_eq5_w_img_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_w_img_pred.bias'])
        model.roi_heads.relation.predictor.fc_eq5_u_img_pred.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_eq5_u_img_pred.bias'])
        model.roi_heads.relation.predictor.fc_output_proj_img_pred.model[0].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_output_proj_img_pred.model.0.linear.bias'])
        model.roi_heads.relation.predictor.fc_output_proj_img_pred.model[2].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_output_proj_img_pred.model.2.linear.bias'])
        model.roi_heads.relation.predictor.fc_output_proj_ont_pred.model[0].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_output_proj_ont_pred.model.0.linear.bias'])
        model.roi_heads.relation.predictor.fc_output_proj_ont_pred.model[2].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_output_proj_ont_pred.model.2.linear.bias'])
        model.roi_heads.relation.predictor.fc_mp_receive_img_ent.model[0].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_img_ent.model.0.linear.bias'])
        model.roi_heads.relation.predictor.fc_mp_receive_img_pred.model[0].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_receive_img_pred.model.0.linear.bias'])
        model.roi_heads.relation.predictor.fc_mp_send_ont_ent.model[0].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_ont_ent.model.0.linear.bias'])
        model.roi_heads.relation.predictor.fc_mp_send_ont_pred.model[0].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_ont_pred.model.0.linear.bias'])
        model.roi_heads.relation.predictor.fc_mp_send_img_ent.model[0].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_img_ent.model.0.linear.bias'])
        model.roi_heads.relation.predictor.fc_mp_send_img_pred.model[0].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_img_pred.model.0.linear.bias'])
        model.roi_heads.relation.predictor.fc_mp_send_ont_ent.model[2].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_ont_ent.model.2.linear.bias'])
        model.roi_heads.relation.predictor.fc_mp_send_ont_pred.model[2].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_ont_pred.model.2.linear.bias'])
        model.roi_heads.relation.predictor.fc_mp_send_img_ent.model[2].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_img_ent.model.2.linear.bias'])
        model.roi_heads.relation.predictor.fc_mp_send_img_pred.model[2].linear.bias.copy_(state_dict['ggnn_rel_reason.ggnn.fc_mp_send_img_pred.model.2.linear.bias'])
    del state_dict
