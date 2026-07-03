from __future__ import annotations

import torch
from lightning import Trainer
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR

def reset_cosine_scheduler(
    trainer: Trainer,
    *,
    lr: float,
    t_max: int,
    eta_min: float = 0.0,
) -> None:
    """Replace the active cosine scheduler with a fresh cycle starting at ``lr``."""
    if not trainer.lr_scheduler_configs:
        return

    scheduler_config = trainer.lr_scheduler_configs[0]
    optimizers = trainer.optimizers
    if not optimizers:
        return
    optimizer = optimizers[0]

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    new_cosine = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    scheduler = scheduler_config.scheduler

    if isinstance(scheduler, SequentialLR):
        scheduler._schedulers[-1] = new_cosine
    else:
        scheduler_config.scheduler = new_cosine


def compute_cosine_t_max(
    *,
    epochs: int,
    steps_per_epoch: int,
    warmup_steps: int,
    subtract_warmup: bool,
) -> int:
    total_steps = epochs * steps_per_epoch
    if subtract_warmup:
        return max(total_steps - warmup_steps, 1)
    return max(total_steps, 1)
