# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)

import argparse
from os.path import join as os_path_join
import os
from time import time as time_time
import datetime
import random
from resource import RLIMIT_NOFILE, getrlimit, setrlimit

import numpy as np
from torch import cat as torch_cat, tensor as torch_tensor, manual_seed as torch_manual_seed, device as torch_device
from torch.cuda import max_memory_allocated, set_device, manual_seed_all
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torch.multiprocessing import set_sharing_strategy

from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex.amp import initialize as amp_initialize, scale_loss as amp_scale_loss
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def setup_seed(seed):
    torch_manual_seed(seed)
    manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

def train(cfg, local_rank, distributed, logger):
    val_before = True # True False
    print_etari = 200 # 3   200
    debug_print(logger, 'prepare training')
    model = build_detection_model(cfg)
    debug_print(logger, 'end model construction')

    clean_classifier = cfg.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER
    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)

    #fix_eval_modules(eval_modules)
    if clean_classifier:
        fix_eval_modules_no_classifier(model, with_grad_name='_clean')
    else:
        fix_eval_modules(eval_modules)

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR in ["IMPPredictor",]:
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor",]
    else:
        slow_heads = []

    device = torch_device(cfg.MODEL.DEVICE)
    model.to(device)

    num_batch = cfg.SOLVER.IMS_PER_BATCH
    optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=0.1,
                                rl_factor=float(num_batch))
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    debug_print(logger, 'end optimizer and shcedule')
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp_initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, 'end distributed')
    arguments = {}
    arguments["iteration"] = 0

    # load pretrain layers to new layers
    load_mapping = {"roi_heads.relation.box_feature_extractor" : "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor" : "roi_heads.box.feature_extractor"}

    if cfg.MODEL.ATTRIBUTE_ON:
        load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        load_mapping["roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    if checkpointer.has_checkpoint():
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT,
                                       update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        arguments.update(extra_checkpoint_data)
    else:
        checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)
        # load_mapping is only used when we init current model from detection model.
        # load base model
        if clean_classifier:
            debug_print(logger, 'end load checkpointer')

            # load_mapping_classifier = {
            #     "roi_heads.relation.predictor.rel_compress_clean":"roi_heads.relation.predictor.rel_compress",
            #     "roi_heads.relation.predictor.ctx_compress_clean": "roi_heads.relation.predictor.ctx_compress",
            #     "roi_heads.relation.predictor.freq_bias_clean": "roi_heads.relation.predictor.freq_bias",
            #     "roi_heads.relation.predictor.post_cat_clean": "roi_heads.relation.predictor.post_cat",
            # }
            if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "TransformerTransferPredictor":
                load_mapping_classifier = {
                    "roi_heads.relation.predictor.rel_compress_clean": "roi_heads.relation.predictor.rel_compress",
                    "roi_heads.relation.predictor.ctx_compress_clean": "roi_heads.relation.predictor.ctx_compress",
                    "roi_heads.relation.predictor.freq_bias_clean": "roi_heads.relation.predictor.freq_bias",
                }
            if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "VCTreePredictor":
                load_mapping_classifier = {
                    "roi_heads.relation.predictor.ctx_compress_clean": "roi_heads.relation.predictor.ctx_compress",
                    "roi_heads.relation.predictor.freq_bias_clean": "roi_heads.relation.predictor.freq_bias",
                    "roi_heads.relation.predictor.post_cat_clean": "roi_heads.relation.predictor.post_cat",
                }
            if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "MotifPredictor":
                load_mapping_classifier = {
                    "roi_heads.relation.predictor.rel_compress_clean": "roi_heads.relation.predictor.rel_compress",
                    "roi_heads.relation.predictor.freq_bias_clean": "roi_heads.relation.predictor.freq_bias",
                    "roi_heads.relation.predictor.post_cat_clean": "roi_heads.relation.predictor.post_cat",
                }
            #load_mapping_classifier = {}
            if cfg.MODEL.PRETRAINED_MODEL_CKPT != "" :
                debug_print(logger, 'load PRETRAINED_MODEL_CKPT!!!!')
                checkpointer.load(cfg.MODEL.PRETRAINED_MODEL_CKPT, update_schedule=False,
                                 with_optim=False, load_mapping=load_mapping_classifier)
    # debug_print(logger, 'load PRETRAINED_MODEL_CKPT!!!!')
    # checkpointer.load(cfg.MODEL.PRETRAINED_MODEL_CKPT, update_schedule=False,
    #                  with_optim=False)
    debug_print(logger, 'end load checkpointer')
    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    val_data_loaders = make_data_loader(
        cfg,
        mode='val',
        is_distributed=distributed,
    )
    debug_print(logger, 'end dataloader')
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    writer = SummaryWriter(log_dir=os_path_join(output_dir, 'tensorboard'))
    if cfg.SOLVER.PRE_VAL and val_before:
        logger.info("Validate before training")
        run_val(cfg, model, val_data_loaders, distributed, logger, writer, 0, output_dir)

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    logger.info("Number iteration: "+str(max_iter))
    start_iter = arguments["iteration"]
    start_training_time = time_time()
    end = time_time()
    print_first_grad = True
    mode = None
    use_gt_box = cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX
    use_gt_object_label = cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
    if use_gt_box is True and use_gt_object_label is True:
        mode = 'predcls'
    if use_gt_box is True and use_gt_object_label is False:
        mode = 'sgcls'
    if use_gt_box is False and use_gt_object_label is False:
        mode = 'sgdet'
    if mode is None:
        raise ValueError(f'mode is None given use_gt_box={use_gt_box} and use_gt_object_label={use_gt_object_label}')
    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
        # if iteration % 1000 == 0:
        #     haaa=0
        if any(len(target) < 1 for target in targets):
            logger.error("Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}")
        data_time = time_time() - end
        iteration += 1
        arguments["iteration"] = iteration

        model.train()
        #fix_eval_modules(eval_modules)
        if clean_classifier:
            fix_eval_modules_no_classifier(model, with_grad_name='_clean')
        else :
            fix_eval_modules(eval_modules)

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp_scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()

        # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
        verbose = (iteration % cfg.SOLVER.PRINT_GRAD_FREQ) == 0 or print_first_grad # print grad or not
        print_first_grad = False
        clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad], max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, writer=writer, iteration=iteration, clip=True)

        optimizer.step()

        batch_time = time_time() - end
        end = time_time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % print_etari == 0 or iteration == max_iter:
            lr_i = optimizer.param_groups[-1]["lr"]
            mem_i = max_memory_allocated() / 1048576.0
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=lr_i,
                    memory=mem_i,
                )
            )
            writer.add_scalar(f'{mode}/lr', lr_i, iteration)
            writer.add_scalar(f'{mode}/memory', mem_i, iteration)

        if iteration % checkpoint_period == 0 and iteration != max_iter:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        val_result = None # used for scheduler updating
        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
            logger.info("Start validating")
            val_result = run_val(cfg, model, val_data_loaders, distributed, logger, writer, iteration, output_dir)
            logger.info("Validation Result: %.4f" % val_result)

        # scheduler should be called after optimizer.step() in pytorch>=1.1.0
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
            scheduler.step(val_result, epoch=iteration)
            if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                # break
        else:
            scheduler.step()
        writer.add_scalars(f'{mode}/loss', {'loss': losses_reduced, **loss_dict_reduced}, iteration)
        writer.add_scalars(f'{mode}/time', {'time_batch': batch_time, 'time_data': data_time}, iteration)

    total_training_time = time_time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    writer.close()
    return model

def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False

def fix_eval_modules_no_classifier(module, with_grad_name='_clean'):
    #for module in eval_modules:
    for name, param in module.named_parameters():
        if with_grad_name not in name:
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False

def run_val(cfg, model, val_data_loaders, distributed, logger, writer, iteration, output_dir):
    # set_sharing_strategy('file_system')
    if distributed:
        model = model.module
    #torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )

    dataset_names = cfg.DATASETS.VAL
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_result = inference(
                            cfg,
                            model,
                            val_data_loader,
                            dataset_name=dataset_name,
                            iou_types=iou_types,
                            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=os_path_join(output_dir, dataset_name),
                            logger=logger,
                            writer=writer,
                            iteration=iteration,
                        )
                        # return_all=return_all,
        synchronize()
        val_result.append(dataset_result)
    # support for multi gpu distributed testing
    gathered_result = all_gather(torch_tensor(dataset_result).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch_cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result>=0]
    del gathered_result
    # from evaluate: float(np.mean(result_dict[mode + '_recall'][100]))
    val_result = float(valid_result.mean())
    del valid_result
    #torch.cuda.empty_cache()
    return val_result


def run_test(cfg, model, distributed, logger, iteration):
    model_name = os.environ.get('MODEL_NAME')
    debug_print(logger, f'running val for model {model_name} at iteration={iteration}')
    writer = SummaryWriter(log_dir=os_path_join(cfg.OUTPUT_DIR, 'tensorboard_test'))
    if distributed:
        model = model.module
    #torch.cuda.empty_cache()
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
            output_folder = os_path_join(cfg.OUTPUT_DIR, 'inference_test', dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode='test', is_distributed=distributed)
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
            iteration=iteration,
        )
        synchronize()
    writer.close()


def main():
    setrlimit(RLIMIT_NOFILE, (4096, getrlimit(RLIMIT_NOFILE)[1]))
    setup_seed(20)
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        set_device(args.local_rank)
        init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os_path_join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed, logger)
    model_name = os.environ.get('MODEL_NAME')
    if not args.skip_test:
        run_test(cfg, model, args.distributed, logger, cfg.SOLVER.MAX_ITER)
        logger.info(f'Finished testing model {model_name} at a total of {cfg.SOLVER.MAX_ITER}')

    logger.info(f'Finished training model {model_name} at a total of {cfg.SOLVER.MAX_ITER} iterations')


if __name__ == "__main__":
    main()
