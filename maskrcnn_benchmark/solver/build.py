# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch.optim import SGD
from torch.distributed.optim import ZeroRedundancyOptimizer
from .lr_scheduler import WarmupMultiStepLR, WarmupReduceLROnPlateau


OPTIMIZERS_WITH_SCHEDULERS = {'SGD', 'FusedSGD'}


def make_optimizer(cfg, model, logger, slow_heads=None, slow_ratio=5.0, rl_factor=1.0,
                   wofinetune_params=None, finetune_rate=100, return_lrs_by_name=False):
    optimizer_type = cfg.SOLVER.TYPE
    lrs_by_name = {}
    if optimizer_type in OPTIMIZERS_WITH_SCHEDULERS:
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            if slow_heads is not None:
                for item in slow_heads:
                    if item in key:
                        logger.info("SLOW HEADS: {} is slow down by ratio of {}.".format(key, str(slow_ratio)))
                        lr = lr / slow_ratio
                        break
            if wofinetune_params is not None:
                if wofinetune_params in key:
                    lr = lr
                else:
                    lr = lr / finetune_rate
            lr_final = lr * rl_factor
            lrs_by_name[key] = lr_final
            params += [{"params": [value], "lr": lr_final, "weight_decay": weight_decay}]
            logger.info("params {} lr: {}.".format(key, str(lr_final)))
    else:
        fc_params = []
        non_fc_params = []
        lr = cfg.SOLVER.BASE_LR
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'fc6' in n or 'fc7' in n:
                fc_params.append(p)
                lrs_by_name[key] = lr / 10.0
            else:
                non_fc_params.append(p)
                lrs_by_name[key] = lr
        params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]

    if optimizer_type == 'SGD':
        optimizer_class = SGD

    optimizer = ZeroRedundancyOptimizer(
        params,
        optimizer_class=optimizer_class,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.MOMENTUM,
    )

    if return_lrs_by_name:
        return optimizer, lrs_by_name
    return optimizer


def make_lr_scheduler(cfg, optimizer, logger=None):
    if cfg.SOLVER.SCHEDULE.TYPE == "WarmupMultiStepLR":
        return WarmupMultiStepLR(# TEMP:
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )

    elif cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
        return WarmupReduceLROnPlateau(
            optimizer,
            cfg.SOLVER.SCHEDULE.FACTOR,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            patience=cfg.SOLVER.SCHEDULE.PATIENCE,
            threshold=cfg.SOLVER.SCHEDULE.THRESHOLD,
            cooldown=cfg.SOLVER.SCHEDULE.COOLDOWN,
            logger=logger,
        )

    else:
        raise ValueError("Invalid Schedule Type")
