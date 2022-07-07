# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from apex.optimizers import FusedAdam, FusedSGD
from torch.optim import Adam, SGD
from .lr_scheduler import WarmupMultiStepLR, WarmupReduceLROnPlateau
from .adabound import AdaBound


OPTIMIZERS_WITH_SCHEDULERS = {'SGD', 'FusedSGD'}


def make_optimizer(cfg, model, logger, slow_heads=None, slow_ratio=5.0, rl_factor=1.0,
                   wofinetune_params=None, finetune_rate=100):
    optimizer_type = cfg.SOLVER.TYPE
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
            params += [{"params": [value], "lr": lr * rl_factor, "weight_decay": weight_decay}]
            logger.info("params {} lr: {}.".format(key, str(lr * rl_factor)))
    else:
        fc_params = []
        non_fc_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'fc6' in n or 'fc7' in n:
                fc_params.append(p)
            else:
                non_fc_params.append(p)
        # fc_params = [p for n,p in model.named_parameters() if n.startswith('fc') and p.requires_grad]
        # non_fc_params = [p for n,p in model.named_parameters() if not n.startswith('fc') and p.requires_grad]
        lr = cfg.SOLVER.BASE_LR
        params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]
    if optimizer_type == 'Adam':
        return Adam(params, lr=cfg.SOLVER.BASE_LR, eps=1e-3) # From GB-Net
    if optimizer_type == 'FusedAdam':
        return FusedAdam(params, lr=cfg.SOLVER.BASE_LR, eps=1e-3) # From GB-Net
    if optimizer_type == 'SGD':
        return SGD(params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    if optimizer_type == 'FusedSGD':
        return FusedSGD(params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    if optimizer_type == 'AdaBound':
        return AdaBound(params, lr=cfg.SOLVER.BASE_LR)


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
