# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import cat as torch_cat, full as torch_full, nonzero as torch_nonzero, zeros as torch_zeros, log2 as torch_log2, tensor as torch_tensor, sqrt as torch_sqrt, floor as torch_floor, clamp as torch_clamp, float32 as torch_float32, int64 as torch_int64
from torch.nn import Module, ModuleList

from maskrcnn_benchmark.layers import ROIAlign
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3

from .utils import cat


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch_sqrt(cat([boxlist.area() for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch_floor(self.lvl0 + torch_log2(s / self.s0 + self.eps))
        target_lvls = torch_clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch_int64) - self.k_min


class Pooler(Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """
    # NOTE: cat_all_levels is added for relationship detection. We want to concatenate
    # all levels, since detector is fixed in relation detection. Without concatenation
    # if there is any difference among levels, it can not be finetuned anymore.
    def __init__(self, output_size, scales, sampling_ratio, in_channels=512, cat_all_levels=False):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
                )
            )
        self.poolers = ModuleList(poolers)
        # (Pdb) self.poolers
        # ModuleList(
        #   (0): ROIAlign(output_size=(7, 7), spatial_scale=0.25, sampling_ratio=2)
        #   (1): ROIAlign(output_size=(7, 7), spatial_scale=0.125, sampling_ratio=2)
        #   (2): ROIAlign(output_size=(7, 7), spatial_scale=0.0625, sampling_ratio=2)
        #   (3): ROIAlign(output_size=(7, 7), spatial_scale=0.03125, sampling_ratio=2)
        # )

        self.output_size = output_size
        self.cat_all_levels = cat_all_levels
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -torch_log2(torch_tensor(scales[0], dtype=torch_float32)).item()
        lvl_max = -torch_log2(torch_tensor(scales[-1], dtype=torch_float32)).item()
        self.map_levels = LevelMapper(lvl_min, lvl_max)
        # reduce the channels
        if self.cat_all_levels:
            self.reduce_channel = make_conv3x3(in_channels * len(self.poolers), in_channels, dilation=1, stride=1, use_relu=True)


    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [
                torch_full((len(b), 1), i, dtype=dtype, device=device)
                for i, b in enumerate(boxes)
            ],
            dim=0,
        )
        return torch_cat([ids, concat_boxes], dim=1)

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
                                (Pdb) len(x)
                                5
                                (Pdb) x[0].size()
                                torch.Size([16, 256, 256, 152])

            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
                                (Pdb) len(boxes)
                                16
                                (Pdb) boxes[0]
                                BoxList(num_boxes=1007, image_width=600, image_height=900, mode=xyxy)
                                (Pdb) boxes[0].bbox.size()
                                torch.Size([1007, 4])
                                (Pdb) boxes[0].bbox.size()
                                torch.Size([80, 4])


        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        rois = self.convert_to_roi_format(boxes)
        assert rois.size(0) > 0
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(boxes)

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        final_channels = num_channels * num_levels if self.cat_all_levels else num_channels
        result = torch_zeros(
            (num_rois, final_channels, output_size, output_size),
            dtype=dtype,
            device=device,
        )
        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            if self.cat_all_levels: # False
                result[:,level*num_channels:(level+1)*num_channels,:,:] = pooler(per_level_feature, rois).to(dtype)
            else:
                idx_in_level = torch_nonzero(levels == level).squeeze(1)
                rois_per_level = rois[idx_in_level]
                result[idx_in_level] = pooler(per_level_feature, rois_per_level).to(dtype)
        if self.cat_all_levels: # False
            return self.reduce_channel(result)
        return result


def make_pooler(cfg, head_name):
    resolution = cfg.MODEL[head_name].POOLER_RESOLUTION
    scales = cfg.MODEL[head_name].POOLER_SCALES
    sampling_ratio = cfg.MODEL[head_name].POOLER_SAMPLING_RATIO
    pooler = Pooler(
        output_size=(resolution, resolution),
        scales=scales,
        sampling_ratio=sampling_ratio,
    )
    return pooler
