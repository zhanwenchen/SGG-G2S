# -*- coding: utf-8 -*-
from os import environ as os_environ
from os.path import join as os_path_join, dirname as os_path_dirname
from json import load as json_load
from pickle import load as pickle_load
from numpy import array as np_array
from torch import (
    Tensor,
    no_grad as torch_no_grad,
    randperm as torch_randperm,
    stack as torch_stack,
    from_numpy as torch_from_numpy,
    topk as torch_topk,
    reciprocal as torch_reciprocal,
    cat as torch_cat,
    sort as torch_sort,
    as_tensor as torch_as_tensor,
    empty as torch_empty,
    isnan as torch_isnan,
    nan_to_num as torch_nan_to_num,
)
from maskrcnn_benchmark.structures.image_list import ImageList
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data import get_dataset_statistics # calls VisualGenome.get_statistics which calls get_VG_statistics


class RelationAugmenter(object):
    # __doc__ = r"""Applies a 1D convolution over an input signal composed of several input
    # planes.
    # In the simplest case, the output value of the layer with input size
    # :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
    # precisely described as:
    # .. math::
    #     \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
    #     \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
    #     \star \text{input}(N_i, k)
    # where :math:`\star` is the valid `cross-correlation`_ operator,
    # :math:`N` is a batch size, :math:`C` denotes a number of channels,
    # :math:`L` is a length of signal sequence.
    # """ + r"""
    # This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    # On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.
    # * :attr:`stride` controls the stride for the cross-correlation, a single
    #   number or a one-element tuple.
    # * :attr:`padding` controls the amount of padding applied to the input. It
    #   can be either a string {{'valid', 'same'}} or a tuple of ints giving the
    #   amount of implicit padding applied on both sides.
    # * :attr:`dilation` controls the spacing between the kernel points; also
    #   known as the à trous algorithm. It is harder to describe, but this `link`_
    #   has a nice visualization of what :attr:`dilation` does.
    # {groups_note}
    # Note:
    #     {depthwise_separable_note}
    # Note:
    #     {cudnn_reproducibility_note}
    # Note:
    #     ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
    #     the input so the output has the shape as the input. However, this mode
    #     doesn't support any stride values other than 1.
    # Note:
    #     This module supports complex data types i.e. ``complex32, complex64, complex128``.
    # Args:
    #     in_channels (int): Number of channels in the input image
    #     out_channels (int): Number of channels produced by the convolution
    #     kernel_size (int or tuple): Size of the convolving kernel
    #     stride (int or tuple, optional): Stride of the convolution. Default: 1
    #     padding (int, tuple or str, optional): Padding added to both sides of
    #         the input. Default: 0
    #     padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
    #         ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
    #     dilation (int or tuple, optional): Spacing between kernel
    #         elements. Default: 1
    #     groups (int, optional): Number of blocked connections from input
    #         channels to output channels. Default: 1
    #     bias (bool, optional): If ``True``, adds a learnable bias to the
    #         output. Default: ``True``
    # """.format(**reproducibility_notes, **convolution_notes) + r"""
    # Shape:
    #     - Input: :math:`(N, C_{in}, L_{in})` or :math:`(C_{in}, L_{in})`
    #     - Output: :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`, where
    #       .. math::
    #           L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
    #                     \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor
    # Attributes:
    #     weight (Tensor): the learnable weights of the module of shape
    #         :math:`(\text{out\_channels},
    #         \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size})`.
    #         The values of these weights are sampled from
    #         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
    #         :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`
    #     bias (Tensor):   the learnable bias of the module of shape
    #         (out_channels). If :attr:`bias` is ``True``, then the values of these weights are
    #         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
    #         :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`
    # Examples::
    #     >>> m = nn.Conv1d(16, 33, 3, stride=2)
    #     >>> input = torch.randn(20, 16, 50)
    #     >>> output = m(input)
    # .. _cross-correlation:
    #     https://en.wikipedia.org/wiki/Cross-correlation
    # .. _link:
    #     https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    # """
    def __init__(self, pred_counts, bottom_k: int, strategy: str, cfg=None):
        DATA_DIR = os_environ['DATA_DIR_VG_RCNN']
        with open(os_path_join(DATA_DIR, 'visual_genome', 'VG-SGG-dicts-with-attri.json'), 'r') as fin:
            scene_graph_meta = json_load(fin)
        self.idx2preds = ['_'] + list(scene_graph_meta['idx_to_predicate'].values())
        self.idx2preds_np = np_array(self.idx2preds)

        n = len(pred_counts)
        pred_counts_minus_1 = pred_counts[1:] # Exclude the background pred.

        if bottom_k > -1:
            _, pred_counts_sorted_indices = torch_sort(pred_counts_minus_1, descending=False)
            print(f'Only augment the {bottom_k} least frequent relations out of {n}')
            # _, indices = torch_topk(P_REL_ALL_INV, bottom_k, largest=True, sorted=True)
            # don't dynamically sort.
            pred_counts_sorted_indices_bottom = pred_counts_sorted_indices[:bottom_k] + 1 # added back
            pred_counts_sorted_indices_bottom = pred_counts_sorted_indices_bottom.tolist()
            self.bottom_k_rels = set(pred_counts_sorted_indices_bottom)
            pred_counts_sorted_indices_top = pred_counts_sorted_indices[bottom_k:] + 1 # added back
        else:
            self.bottom_k_rels = None

        if strategy == 'random':
            # Construct the inverse relation frequency distribution
            pred_counts_dist = pred_counts_minus_1/pred_counts_minus_1.sum() # population dist
            # Or we can assume a gaussian dist., etc.
            p_rel_all_inv = torch_reciprocal(pred_counts_dist)
            p_rel_all_inv = torch_cat((torch_as_tensor([0]), p_rel_all_inv))
            p_rel_all_inv[pred_counts_sorted_indices_top] = 0 # set highest to 0
            p_rel_all_inv /= p_rel_all_inv.sum()

            p_rel_all_inv_cache = p_rel_all_inv.repeat(n, 1)
            # Cache
            p_rel_all_inv_cache.fill_diagonal_(0)
            self.dist_rels_all_excluded_by = p_rel_all_inv_cache
        elif strategy == 'cooccurrence-pred_cov':
            all_edges_fpath = os_environ['ALL_EDGES_FPATH']
            with open(all_edges_fpath, 'rb') as f:
                all_edges = pickle_load(f)

            edges_ent2ent = all_edges['edges_ent2ent']
            print('edges_ent2ent.shape =', edges_ent2ent.shape)
            p_obj_given_subj_zareian = edges_ent2ent[5, :, :].T
            p_subj_given_obj_zareian = edges_ent2ent[6, :, :].T
            noun_cov_zareian = edges_ent2ent[7, :, :].T
            edges_wordnet_zareian = edges_ent2ent[8, :, :]

            edges_pred2pred = all_edges['edges_pred2pred']
            print('edges_pred2pred.shape =', edges_pred2pred.shape)
            pred_cov_zareian = edges_pred2pred[3, : :].T

            edges_ent2pred = all_edges['edges_ent2pred']
            print('edges_ent2pred.shape =', edges_ent2pred.shape)
            p_subj_given_pred_zareian = edges_ent2pred[1, :, :].T
            p_obj_given_pred_zareian = edges_ent2pred[2, :, :].T

            edges_pred2ent = all_edges['edges_pred2ent']
            print('edges_pred2ent.shape =', edges_pred2ent.shape)
            p_pred_given_subj_zareian = edges_pred2ent[1, :, :].T
            p_pred_given_obj_zareian = edges_pred2ent[2, :, :].T
            self.cooccurrence = torch_from_numpy(pred_cov_zareian)
            # set top k to 0
            self.cooccurrence[:, pred_counts_sorted_indices_top] = 0
        elif strategy == 'cooccurrence-fgmat':
            statistics = get_dataset_statistics(cfg)
            fg_matrix = statistics['fg_matrix'] # [151, 151, 51]
            fg_matrix[:, :, 0] = 0 # we don't care about background.
            fg_matrix[0, :, :] = 0 # we don't care about background.
            fg_matrix[:, 0, :] = 0 # we don't care about background.
            cooccurrence = fg_matrix / fg_matrix.sum(dim=-1).unsqueeze_(-1)
            self.cooccurrence = torch_nan_to_num(cooccurrence, nan=0.0, out=cooccurrence)
            self.cooccurrence[:, :, pred_counts_sorted_indices_top] = 0
        elif strategy == 'wordnet':
            with open(os_path_join(os_path_dirname(__file__), 'pred_sim.pkl'), 'rb') as f:
                pred_sim_path_sim = pickle_load(f)[0, :, :]
            pred_sim_path_sim = torch_from_numpy(pred_sim_path_sim)
            pred_sim_path_sim.fill_diagonal_(0)
            pred_sim_path_sim /= pred_sim_path_sim.sum(dim=-1).unsqueeze_(-1)
            torch_nan_to_num(pred_sim_path_sim, nan=0.0, out=pred_sim_path_sim)
            pred_sim_path_sim[pred_counts_sorted_indices_top, :] = 0
            pred_sim_path_sim[:, pred_counts_sorted_indices_top] = 0
            self.cooccurrence = pred_sim_path_sim
        elif strategy == 'csk':
            all_edges_fpath = os_environ['ALL_EDGES_FPATH']
            with open(all_edges_fpath, 'rb') as f:
                all_edges = pickle_load(f)

            edges_pred2pred = all_edges['edges_pred2pred']
            # Sum all edge types because we don't care about which one
            cooccurrence = edges_pred2pred[:3, :, :].sum(axis=0)
            cooccurrence = torch_from_numpy(cooccurrence)

            # No need to normalize because multinomial will treat it as weights
            # set top k to 0
            cooccurrence[:, pred_counts_sorted_indices_top] = 0
            cooccurrence.fill_diagonal_(0)
            torch_nan_to_num(cooccurrence, nan=0.0, out=cooccurrence)
            # print(f'csk: cooccurrence.sum() = {cooccurrence.sum()}') # only 19
            self.cooccurrence = cooccurrence
        else:
            raise ValueError(f'Invalid strategy: {strategy}')
        self.strategy = strategy

    @torch_no_grad()
    def sample_random(self, idx_rel: int, num2aug: int, replace: bool, subj=None, obj=None) -> Tensor:
         # r"""
        #     Given a relation, outputs the related relations.
        #
        #     Args:
        #         idx_rel (int): The index of the relation in the sorted global relations.
        #         num2aug (int): The size of each output sample.
        #         replace (bool): Whether to allow duplicates in the output relations.
        #         distribution (tensor): The frequency distribution of the
        #
        #     Shape:
        #         - Input: :math:`(*, H_{in})`, where :math:`*` represents any number of
        #           dimensions (including none) and :math:`H_{in} = \text{in\_features}`.
        #         - Output: :math:`(*, H_{out})`, where all but the last dimension
        #           match the input shape and :math:`H_{out} = \text{out\_features}`.
        #
        #     Attributes:
        #         weight: The learnable weights of the module of shape :math:`(H_{out}, H_{in})`, where
        #                 :math:`H_{in} = \text{in\_features}` and :math:`H_{out} = \text{out\_features}`.
        #                 The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
        #                 :math:`k = \frac{1}{\text{in1\_features}}`.
        #         bias: The learnable bias of the module of shape :math:`(H_{out})`. Only present when
        #               :attr:`bias` is ``True``. The values are initialized from
        #               :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where :math:`k = \frac{1}{H_{in}}`.
        #
        #     Examples::
        #         >>> m = nn.Linear(20, 30)
        #         >>> input = torch.randn(128, 20)
        #         >>> output = m(input)
        #         >>> print(output.size())
        #         torch.Size([128, 30])
        #
        #         >>> # Example of creating a Linear layer with no bias.
        #         >>> m = nn.Linear(3, 3, bias=False)
        #         >>> input = torch.randn(10, 3)
        #         >>> output = m(input)
        #         >>> print(output.size())
        #         torch.Size([10, 3])
        #     """
        #     '''
        #     Given a single triplet in the form of (subj, rel, obj) and outputs a list of triplets with
        #     the , not including the original. You should append the original.
        #
        #     Note that the input indices much be singular (one element) but the outputs will have multiple
        #     (specificall num2aug elements).
        #     '''
        return self.dist_rels_all_excluded_by[idx_rel].multinomial(num2aug, replacement=replace)

    @torch_no_grad()
    def sample_predcov(self, idx_rel: int, num2aug: int, replace: bool, subj=None, obj=None) -> Tensor:
        cooccurrence_current = self.cooccurrence[idx_rel]
        if cooccurrence_current.count_nonzero() > 0:
            return cooccurrence_current.multinomial(num2aug, replacement=replace)
        else:
            return torch_empty(0)

    @torch_no_grad()
    def sample_fgmat(self, idx_rel: int, num2aug: int, replace: bool, subj=None, obj=None) -> Tensor:
        # TODO: for all sampling methods, return empty if idx_rel == 0
        if subj == 0 or obj == 0:
            return torch_empty(0)
        rel_counts = self.cooccurrence[subj, obj, :]
        rel_counts[idx_rel] = 0
        if rel_counts.count_nonzero() > 0 and not torch_isnan(rel_counts).any():
            return rel_counts.multinomial(num2aug, replacement=replace)
        else:
            return torch_empty(0)

    @torch_no_grad()
    def sample_wordnet(self, idx_rel: int, num2aug: int, replace: bool, subj=None, obj=None) -> Tensor:
        cooccurrence_current = self.cooccurrence[idx_rel]
        if cooccurrence_current.count_nonzero() > 0:
            return cooccurrence_current.multinomial(num2aug, replacement=replace)
        else:
            return torch_empty(0)

    @torch_no_grad()
    def sample_csk(self, idx_rel: int, num2aug: int, replace: bool, subj=None, obj=None) -> Tensor:
        cooccurrence_current = self.cooccurrence[idx_rel]
        if cooccurrence_current.count_nonzero() > 0:
            return cooccurrence_current.multinomial(num2aug, replacement=replace)
        else:
            return torch_empty(0)

    @torch_no_grad()
    def augment(self, images, targets, num2aug: int, randmax: int):
        # NOTE: not for cutmix because of rel_new
        # TODO: vectorized
        bottom_k_rels = self.bottom_k_rels
        strategy = self.strategy
        if strategy == 'random':
            sample_func = self.sample_random
        elif strategy == 'cooccurrence-pred_cov':
            sample_func = self.sample_predcov
        elif strategy == 'cooccurrence-fgmat':
            sample_func = self.sample_fgmat
        elif strategy == 'wordnet':
            sample_func = self.sample_wordnet
        elif strategy == 'csk': # Commonsence Knowledge
            sample_func = self.sample_csk
        else:
            raise ValueError(f'Invalid strategy: {strategy}')

        device = targets[0].bbox.device
        images_augmented = []
        image_sizes_augmented = []
        targets_augmented = []

        for image, image_size, target in zip(images.tensors, images.image_sizes, targets):
            relation_old = target.extra_fields['relation']

            # triplets are represented as the relation map.
            idx_subj, idx_obj = idx_rel = relation_old.nonzero(as_tuple=True) # tuple
            rels = relation_old[idx_rel]

            # First add old
            images_augmented.append(image)
            image_sizes_augmented.append(image_size)
            targets_augmented.append(target)

            for idx_subj_og, rel_og, idx_obj_og in zip(idx_subj, rels, idx_obj):
                # QUESTION: Should we augment the bottom og or the bottom others?
                # ANSWER: For cutmix-like, we only augment the bottom rel_og.
                # For others, we save the bottom rel_new.
                rels_new = sample_func(rel_og, num2aug, False, subj=idx_subj_og, obj=idx_obj_og)
                if rels_new.nelement() == 0:
                    print(f'no rels_new for rel_og={self.idx2preds[rel_og]}')
                else:
                    print(f'considering rel_og={self.idx2preds[rel_og]} => rel_new={self.idx2preds_np[rels_new]}')

                for rel_new in rels_new:
                    if bottom_k_rels and int(rel_new) not in bottom_k_rels:
                        continue
                    images_augmented.append(image)
                    image_sizes_augmented.append(image_size)
                    # Triplet to Map
                    relation_new = relation_old.detach().clone()
                    target_new = target.copy_with_fields(target.fields())
                    relation_new[idx_subj_og, idx_subj_og] = rel_new
                    target_new.extra_fields['relation'] = relation_new
                    targets_augmented.append(target_new)
        del images, targets
        if randmax > -1:
            idx_randperm = torch_randperm(len(images_augmented), device=device)[:randmax]
            images_augmented = [images_augmented[i] for i in idx_randperm]
            image_sizes_augmented = [image_sizes_augmented[i] for i in idx_randperm]
            targets_augmented = [targets_augmented[i] for i in idx_randperm]

        return ImageList(torch_stack(images_augmented, dim=0), image_sizes_augmented), targets_augmented

    # @torch_no_grad()
    # def augment_new(self, images, targets, num2aug: int, randmax: int):
    #     # TODO: vectorized
    #     device = targets[0].bbox.device
    #     images_augmented = []
    #     targets_augmented = []
    #
    #     num_images = len(images.image_sizes)
    #
    #     # For each image in the batch
    #     for idx_image, (image, target) in enumerate(zip(images.tensors, targets)):
    #         breakpoint()
    #         image_augmented = []
    #         target_augmented = []
    #         relation_old = target.extra_fields['relation']
    #
    #         # triplets are represented as the relation map.
    #         idx_subj, idx_obj = idx_rel = relation_old.nonzero(as_tuple=True) # tuple
    #         rels = relation_old[idx_rel]
    #         num_rels = len(rels)
    #
    #         # First add old rel
    #         image_augmented.append(image)
    #         target_augmented.append(target)
    #         print(f'augment: processing the {idx_image+1}/{num_images} image of size {images.image_sizes[idx_image]} with {num_rels} old rels')
    #
    #         # expected_len = num_images * num_rels * average(num_new_rels). All rels are in one image.
    #         # But each new rel requires a new image because of single-label
    #         # For each old rel in the image
    #         for idx_subj_og, rel_og, idx_obj_og in zip(idx_subj, rels, idx_obj):
    #             rels_new = self.sample(rel_og, num2aug, True)
    #             for rel_new in rels_new:
    #                 image_augmented.append(image) #
    #
    #                 # Triplet to Map
    #                 relation_new = relation_old.detach().clone()
    #                 # target_new = deepcopy(target) # deepcopy doesn't really create a copy of tensors
    #                 target_new = target.copy_with_fields([]) # deepcopy doesn't really create a copy of tensors
    #                 # print(target_new.size)
    #                 # breakpoint()
    #                 relation_new[idx_subj_og, idx_subj_og] = rel_new
    #                 target_new.extra_fields['relation'] = relation_new
    #                 target_augmented.append(target_new)
    #         if randmax > -1:
    #             idx_randperm = torch_randperm(len(image_augmented), device=device)[:randmax]
    #             image_augmented = [image_augmented[i] for i in idx_randperm]
    #             target_augmented = [target_augmented[i] for i in idx_randperm]
    #         images_augmented.extend(image_augmented)
    #         targets_augmented.extend(target_augmented)
    #     assert len(images_augmented) == len(targets_augmented)
    #     # breakpoint()
    #     return to_image_list(images_augmented), targets_augmented
