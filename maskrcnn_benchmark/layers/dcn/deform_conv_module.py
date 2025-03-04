from math import sqrt as math_sqrt
from torch import Tensor as torch_Tensor, chunk as torch_chunk, sigmoid as torch_sigmoid, cat as torch_cat
from torch.nn import Module, Parameter, Conv2d
from torch.nn.modules.utils import _pair

from .deform_conv_func import deform_conv, modulated_deform_conv


class DeformConv(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=False
    ):
        assert not bias
        super(DeformConv, self).__init__()
        self.with_bias = bias

        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = Parameter(
            torch_Tensor(out_channels, in_channels // self.groups,
                         *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math_sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, offset):
        return deform_conv(input, offset, self.weight, self.stride,
                           self.padding, self.dilation, self.groups,
                           self.deformable_groups)

    def __repr__(self):
        return "".join([
            "{}(".format(self.__class__.__name__),
            "in_channels={}, ".format(self.in_channels),
            "out_channels={}, ".format(self.out_channels),
            "kernel_size={}, ".format(self.kernel_size),
            "stride={}, ".format(self.stride),
            "dilation={}, ".format(self.dilation),
            "padding={}, ".format(self.padding),
            "groups={}, ".format(self.groups),
            "deformable_groups={}, ".format(self.deformable_groups),
            "bias={})".format(self.with_bias),
        ])


class ModulatedDeformConv(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=True
    ):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias

        self.weight = Parameter(torch_Tensor(
            out_channels,
            in_channels // groups,
            *self.kernel_size
        ))
        if bias:
            self.bias = Parameter(torch_Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math_sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input, offset, mask):
        return modulated_deform_conv(
            input, offset, mask, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.deformable_groups)

    def __repr__(self):
        return "".join([
            "{}(".format(self.__class__.__name__),
            "in_channels={}, ".format(self.in_channels),
            "out_channels={}, ".format(self.out_channels),
            "kernel_size={}, ".format(self.kernel_size),
            "stride={}, ".format(self.stride),
            "dilation={}, ".format(self.dilation),
            "padding={}, ".format(self.padding),
            "groups={}, ".format(self.groups),
            "deformable_groups={}, ".format(self.deformable_groups),
            "bias={})".format(self.with_bias),
        ])

class ModulatedDeformConvPack(ModulatedDeformConv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=True):
        super(ModulatedDeformConvPack, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, deformable_groups, bias)

        self.conv_offset_mask = Conv2d(
            self.in_channels // self.groups,
            self.deformable_groups * 3 * self.kernel_size[0] *
            self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        o1, o2, mask = torch_chunk(self.conv_offset_mask(input), 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv(
            input,offset, mask, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.deformable_groups)
