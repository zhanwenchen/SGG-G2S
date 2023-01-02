# -*- coding: utf-8 -*-
from typing import Tuple
from torch import Tensor
from torch import as_tensor as torch_as_tensor
from maskrcnn_benchmark.structures.image_list import ImageList
from maskrcnn_benchmark.structures.bounding_box import BoxList


def rel_aug(images: ImageList, targets: Tuple[BoxList], num_to_aug: int, strategy: str) -> Tuple[ImageList, Tuple[BoxList]]:
    p_rel_all = pred_counts/pred_counts.sum()
    if strategy == 'all':
        return rel_aug_all(images, targets, p_rel_all, num_to_aug)
    else:
        raise NotImplementedError(f'rel_aug: strategy={strategy} is not implemented yet')


def rel_aug_all_triplets(idx_rel: int, num2aug: int, replace: bool, distribution: Tensor) -> Tensor:
    r"""
    Given a relation, outputs the related relations.

    Args:
        idx_rel (int): The index of the relation in the sorted global relations.
        num2aug (int): The size of each output sample.
        replace (bool): Whether to allow duplicates in the output relations.
        distribution (tensor): The frequency distribution of the

    Shape:
        - Input: :math:`(*, H_{in})`, where :math:`*` represents any number of
          dimensions (including none) and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})`, where all but the last dimension
          match the input shape and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: The learnable weights of the module of shape :math:`(H_{out}, H_{in})`, where
                :math:`H_{in} = \text{in\_features}` and :math:`H_{out} = \text{out\_features}`.
                The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in1\_features}}`.
        bias: The learnable bias of the module of shape :math:`(H_{out})`. Only present when
              :attr:`bias` is ``True``. The values are initialized from
              :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where :math:`k = \frac{1}{H_{in}}`.

    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])

        >>> # Example of creating a Linear layer with no bias.
        >>> m = nn.Linear(3, 3, bias=False)
        >>> input = torch.randn(10, 3)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([10, 3])
    """
    '''
    Given a single triplet in the form of (subj, rel, obj) and outputs a list of triplets with
    the , not including the original. You should append the original.

    Note that the input indices much be singular (one element) but the outputs will have multiple
    (specificall num2aug elements).
    '''
    # Construct the inverse relation frequency distribution
    n = len(pred_counts)
    P_REL_ALL = 1 - (pred_counts/pred_counts.sum()).repeat(n, 1)
    DIST_RELS_ALL_EXCLUDED_BY = P_REL_ALL.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1) # remove diagonal values

    return DIST_RELS_ALL_EXCLUDED_BY[idx_rel].multinomial(num2aug, replacement=replace)


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
        P_REL_ALL = 1.0 - (pred_counts/pred_counts.sum()).repeat(n, 1)
        # Cache
        self.dist_rels_all_excluded_by = P_REL_ALL.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)

    def sample(self, idx_rel, num2aug):
        return self.dist_rels_all_excluded_by[idx_rel].multinomial(num2aug, replacement=replace)

    def augment(self, images, targets):
        # TODO: vectorize
        images_augmented = []
        targets_augmented = []
        for image, target in zip(images.tensors, targets):
            relation_old = target.extra_fields['relation']

            # triplets are represented as the relation map.
            idx_subj, idx_obj = idx_rel = relation_old.nonzero(as_tuple=True) # tuple
            rels = relation_old[idx_rel]

            # First add old
            images_augmented.append(image)
            targets_augmented.append(target)

            for idx_subj_og, rel_og, idx_obj_og in zip(idx_subj, rels, idx_obj):
                rels_new = self.rel_aug_all_triplets(rel_og, 10, True)
                for rel_new in rels_new:
                    images_augmented.append(image)

                    # Triplet to Map
                    relation_new = relation_old.detach().clone()
                    target_new = deepcopy(target)
                    relation_new[idx_subj_og, idx_subj_og] = rel_new
                    target_new.extra_fields['relation'] = relation_new
                    targets_augmented.append(target_new)
        return to_image_list(images_augmented), targets_augmented
