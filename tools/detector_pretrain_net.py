# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
from os import environ as os_environ
from os.path import join as os_path_join
from time import time as time_time
from datetime import timedelta
from random import seed as random_seed
from torch import (
    cat as torch_cat,
    manual_seed as torch_manual_seed,
    device as torch_device,
    as_tensor as torch_as_tensor,
)
from numpy.random import seed as np_random_seed
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.distributed import init_process_group
from torch.cuda import max_memory_allocated, set_device, manual_seed_all, empty_cache
from torch.cuda.amp import autocast, GradScaler
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger


APEX_FUSED_OPTIMIZERS = {'FusedSGD', 'FusedAdam'}
OPTIMIZERS_WITH_SCHEDULERS = {'SGD', 'FusedSGD'}
convert_sync_batchnorm = SyncBatchNorm.convert_sync_batchnorm


# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
# try:
#     from apex import amp
# except ImportError:
#     raise ImportError('Use APEX for multi-precision via apex.amp')
def setup_seed(seed):
    torch_manual_seed(seed)
    manual_seed_all(seed)
    np_random_seed(seed)
    random_seed(seed)
    cudnn.deterministic = True


def train(cfg, local_rank, distributed, logger):
    mode = 'pretrain'
    model = build_detection_model(cfg)
    device = torch_device(cfg.MODEL.DEVICE)
    model.to(device, non_blocking=True)

    using_scheduler = cfg.SOLVER.TYPE in OPTIMIZERS_WITH_SCHEDULERS
    optimizer = make_optimizer(cfg, model, logger, rl_factor=float(cfg.SOLVER.IMS_PER_BATCH))
    scheduler = make_lr_scheduler(cfg, optimizer, logger) if using_scheduler else None

    # Initialize mixed-precision training
    # use_mixed_precision = cfg.DTYPE == "float16"
    # amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    # model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True, # Should be True with a new model
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
    arguments.update(extra_checkpoint_data)

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

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    writer = SummaryWriter(log_dir=os_path_join(output_dir, 'tensorboard_train'))
    if cfg.SOLVER.PRE_VAL:
        logger.info("Validate before training")
        run_val(cfg, model, val_data_loaders, distributed, logger, writer, 0, output_dir)
        logger.info("Finished validation before training")

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time_time()
    end = time_time()
    val_results = []
    scaler = GradScaler()

    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
        model.train()

        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
        data_time = time_time() - end
        iteration += 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device, non_blocking=True)
        targets = [target.to(device) for target in targets]

        if cfg.SOLVER.TYPE in APEX_FUSED_OPTIMIZERS:
            optimizer.zero_grad() # For Apex FusedSGD, FusedAdam, etc
        else:
            optimizer.zero_grad(set_to_none=True) # For Apex FusedSGD, FusedAdam, etc

        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), cfg.SOLVER.GRAD_NORM_CLIP)
        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)

        scale_before = scaler.get_scale()
        # Updates the scale for next iteration.
        scaler.update()

        scale_after = scaler.get_scale()
        skip_lr_sched = scale_before > scale_after
        if skip_lr_sched:
            logger.info(f'i={iteration}: Skipping scheduler scale_before={scale_before}, scale_after={scale_after}')

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision
        batch_time = time_time() - end
        end = time_time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(timedelta(seconds=int(eta_seconds)))

        if iteration % 200 == 0 or iteration == max_iter:
            lr_i = optimizer.param_groups[0]["lr"]
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

        val_result = None # used for scheduler updating
        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
            logger.info("Start validating")
            val_result = run_val(cfg, model, val_data_loaders, distributed, logger, writer, iteration, output_dir)
            logger.info("Validation Result: %.4f" % val_result)
            val_results.append(val_result)

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_{:07d}_final".format(iteration), **arguments)

        if using_scheduler and not skip_lr_sched:
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
    total_time_str = str(timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    writer.close()
    return model


def run_val(cfg, model, val_data_loaders, distributed, logger, writer, iteration, output_dir):
    if distributed:
        model = model.module
    empty_cache()  # TODO check if it helps
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
            output_folder=os_path_join(output_dir, 'inference_val', dataset_name),
            logger=logger,
            writer=writer,
            iteration=iteration,
        )
        synchronize()
        val_result.append(dataset_result)

    gathered_result = all_gather(torch_as_tensor(dataset_result).cpu())
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
    model_name = os_environ.get('MODEL_NAME')
    writer = SummaryWriter(log_dir=os_path_join(cfg.OUTPUT_DIR, 'tensorboard_test'))
    if distributed:
        model = model.module
    empty_cache()  # TODO check if it helps
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
            output_folder = os_path_join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(
        cfg,
        mode='test',
        is_distributed=distributed
        )
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
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
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

    num_gpus = int(os_environ["WORLD_SIZE"]) if "WORLD_SIZE" in os_environ else 1
    distributed = num_gpus > 1

    local_rank = int(os_environ['LOCAL_RANK'])
    if distributed:
        set_device(local_rank)
        init_process_group(
            backend="nccl", init_method="env://",
            timeout=timedelta(seconds=1800),
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

    model = train(cfg, train, distributed, logger)

    if not args.skip_test:
        run_test(cfg, model, distributed, logger, cfg.SOLVER.MAX_ITER)


if __name__ == "__main__":
    main()
