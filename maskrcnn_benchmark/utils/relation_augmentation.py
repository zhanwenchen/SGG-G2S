# -*- coding: utf-8 -*-
from copy import deepcopy
from torch import (
    Tensor,
    no_grad as torch_no_grad,
    randperm as torch_randperm,
    stack as torch_stack,
)
from maskrcnn_benchmark.structures.image_list import ImageList
from maskrcnn_benchmark.structures.bounding_box import BoxList


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
    def __init__(self, pred_counts):
        n = len(pred_counts)
        # Construct the inverse relation frequency distribution

        P_REL_ALL = 1.0 - (pred_counts/pred_counts.sum()).repeat(n, 1)
        # Cache
        self.dist_rels_all_excluded_by = P_REL_ALL.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)

    @torch_no_grad()
    def sample(self, idx_rel: int, num2aug: int, replace: bool) -> Tensor:
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
    def augment(self, images, targets, num2aug: int, randmax: int):
        # TODO: vectorized
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
                rels_new = self.sample(rel_og, num2aug, True)
                for rel_new in rels_new:
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
