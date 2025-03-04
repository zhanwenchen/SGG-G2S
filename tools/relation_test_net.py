# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
from os.path import join as os_path_join
import torch
from torch.utils.tensorboard import SummaryWriter
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank(), filename="log_test.txt")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)


    # load_init_model_name = 'transformer_predcls_dist10_2k_FixPModel_CleanH_Lr1e3_B16'
    # load_mapping_classifier = {
    #     "roi_heads.relation.predictor.rel_compress_clean_10":"roi_heads.relation.predictor.rel_compress_clean",
    #     "roi_heads.relation.predictor.ctx_compress_clean_10": "roi_heads.relation.predictor.ctx_compress_clean",
    #     "roi_heads.relation.predictor.freq_bias_clean_10": "roi_heads.relation.predictor.freq_bias_clean",
    # }
    # #load_mapping_classifier = {}
    # checkpointer.load('checkpoints/' + load_init_model_name + '/model_0014000.pth',
    #                    load_mapping=load_mapping_classifier)
    #
    # load_init_model_name = 'transformer_predcls_dist15_2k_FixPModel_CleanH_Lr1e3_B16'
    # load_mapping_classifier = {
    #     "roi_heads.relation.predictor.rel_compress_clean_15":"roi_heads.relation.predictor.rel_compress_clean",
    #     "roi_heads.relation.predictor.ctx_compress_clean_15": "roi_heads.relation.predictor.ctx_compress_clean",
    #     "roi_heads.relation.predictor.freq_bias_clean_15": "roi_heads.relation.predictor.freq_bias_clean",
    # }
    # #load_mapping_classifier = {}
    # checkpointer.load('checkpoints/' + load_init_model_name + '/model_0004000.pth',
    #                    load_mapping=load_mapping_classifier)
    #
    # load_init_model_name = 'transformer_predcls_dist20_2k_FixPModel_CleanH_Lr1e3_B16'
    # load_mapping_classifier = {
    #     "roi_heads.relation.predictor.rel_compress_clean_20":"roi_heads.relation.predictor.rel_compress_clean",
    #     "roi_heads.relation.predictor.ctx_compress_clean_20": "roi_heads.relation.predictor.ctx_compress_clean",
    #     "roi_heads.relation.predictor.freq_bias_clean_20": "roi_heads.relation.predictor.freq_bias_clean",
    # }
    # #load_mapping_classifier = {}
    # checkpointer.load('checkpoints/' + load_init_model_name + '/model_0010000.pth',
    #                    load_mapping=load_mapping_classifier)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    if cfg.TEST.VAL_FLAG:
        data_loaders_val = make_data_loader(cfg, mode="val", is_distributed=distributed)
    else:
        data_loaders_val = make_data_loader(cfg, mode="test", is_distributed=distributed)
    writer = SummaryWriter(log_dir=os_path_join(output_dir, 'tensorboard_test'))
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
            writer=writer,
            iteration=0,
        )
        synchronize()


if __name__ == "__main__":
    main()
