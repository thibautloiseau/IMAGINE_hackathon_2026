import os
from typing import Dict, List

import hydra
from codecarbon import EmissionsTracker
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import pylogger
from src.utils.codecarbon_utils import configure_codecarbon_gpu_ids

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> Dict[str, Callback]:
    """Instantiates callbacks from config.

    Note: does not allow duplicate callback types!

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A dict of instantiated callbacks, indexed by callback class name.
    """
    callbacks: Dict[str, Callback] = {}

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            cb_name = cb_conf._target_.split(".")[-1]
            callbacks[cb_name] = hydra.utils.instantiate(cb_conf)

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def instantiate_emissions_tracker(cfg: DictConfig) -> EmissionsTracker:
    configure_codecarbon_gpu_ids(cfg)

    os.makedirs(cfg.codecarbon.output_dir, exist_ok=True)

    electricitymaps_api_key = None
    if os.path.isfile(cfg.paths.electricity_maps_key):
        with open(cfg.paths.electricity_maps_key) as f:
            electricitymaps_api_key = f.read().strip()
        log.info("Using Electricity Maps for live carbon intensity tracking")

    tracker: EmissionsTracker = hydra.utils.instantiate(
        cfg.codecarbon, electricitymaps_api_token=electricitymaps_api_key
    )

    return tracker
