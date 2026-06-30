import contextlib
import math
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from codecarbon import EmissionsTracker
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStoppingReason
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch.nn.attention import SDPBackend, sdpa_kernel

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_emissions_tracker,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.progressive_resolution import estimate_training_steps

log = RankedLogger(__name__, rank_zero_only=True)


# Uses TensorFloat32 or bfloat16 for matrix multiplication when available
torch.set_float32_matmul_precision("high")


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    if "CosineAnnealingLR" in cfg.module["main_scheduler"]["_target_"]:
        datamodule.setup(stage="fit")  # Load training set
        callbacks_cfg = cfg.get("callbacks") or {}
        prog_cb = callbacks_cfg.get("progressive_resolution")
        if prog_cb and prog_cb.get("match_tokens"):
            total_steps = estimate_training_steps(
                num_train_samples=len(datamodule.data_train),
                max_epochs=cfg.trainer.max_epochs,
                start_crop_size=cfg.datamodule.train_crop_size,
                full_crop_size=prog_cb.full_crop_size,
                mode=prog_cb.mode,
                step_size=prog_cb.get("step_size", 16),
                epochs_per_stage=prog_cb.get("epochs_per_stage", 1),
                switch_epoch=prog_cb.get("switch_epoch"),
                reference_crop_size=prog_cb.full_crop_size,
                reference_batch_size=prog_cb.reference_batch_size,
            )
        else:
            bsize = cfg.datamodule.batch_size
            total_steps = math.ceil(len(datamodule.data_train) / bsize) * cfg.trainer.max_epochs
        cfg.module.main_scheduler.T_max = total_steps - cfg.module.warmup_steps

    log.info(f"Instantiating module <{cfg.module._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.module)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callback_dict: Dict[str, Callback] = instantiate_callbacks(cfg.get("callbacks"))
    callbacks: List[Callback] = list(callback_dict.values())

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if "codecarbon" in cfg:
        log.info("Instantiating CodeCarbon tracker...")
        tracker: EmissionsTracker = instantiate_emissions_tracker(cfg)
    else:
        tracker = contextlib.nullcontext()

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
        "tracker": tracker,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting training!")
    with tracker, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        # Note: Flash Attention only works with mixed precision, otherwise you will see:
        #   RuntimeError('No available kernel. Aborting execution.')
        # when calling `scaled_dot_product_attention`
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # Check why training stopped
    early_stopping_cb = callback_dict.get("EarlyStopping")

    if early_stopping_cb:
        if early_stopping_cb.stopping_reason == EarlyStoppingReason.STOPPING_THRESHOLD:
            print("Training stopped due to reaching stopping threshold")
        elif early_stopping_cb.stopping_reason == EarlyStoppingReason.NOT_STOPPED:
            print("Training completed normally without early stopping")

        # Access human-readable message
        if early_stopping_cb.stopping_reason_message:
            print(f"Details: {early_stopping_cb.stopping_reason_message}")

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value


if __name__ == "__main__":
    main()
