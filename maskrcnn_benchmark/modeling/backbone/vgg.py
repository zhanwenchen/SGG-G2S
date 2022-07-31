# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
OR:
    model = ResNet(
        "StemWithGN",
        "BottleneckWithGN",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
from torch.nn import Module
from maskrcnn_benchmark.modeling.backbone.load_vgg import load_vgg


class VGG16(Module):
    def __init__(self, cfg):
        super().__init__()
        # vgg = vgg16(pretrained=True)
        self.conv_body = load_vgg().features
        # self.conv_body = Sequential(*list(vgg.features._modules.values())[:-1])

    def forward(self, x):
        return [self.conv_body(x)]
