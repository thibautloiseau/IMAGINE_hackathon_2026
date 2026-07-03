from pathlib import Path
from typing import Optional

import hydra
from lightning import Callback
from omegaconf import DictConfig, open_dict

from compress_imagenet import (
    compressed_datasets_exist,
    compress_splits,
    jpeg_only_dir_name,
    resized_dir_name,
)
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def setup_compressed_data_training(cfg: DictConfig) -> Optional[Callback]:
    """Configure datasets and return the phase-switch callback when enabled."""
    if not cfg.data.get("enable", False):
        return None

    apply_compressed_data_config(cfg)
    prepare_compressed_datasets(cfg)
    return hydra.utils.instantiate(cfg.data.phase_switch_callback)


def apply_compressed_data_config(cfg: DictConfig) -> None:
    """Derive datamodule directories and two-phase settings from ``cfg.data``."""
    data_cfg = cfg.data
    resize_size = data_cfg.resize_size
    crop_size = data_cfg.crop_size
    jpeg_quality = data_cfg.jpeg_quality
    no_jpeg = data_cfg.get("no_jpeg", False)

    phase1_train = resized_dir_name("train", resize_size, crop_size, jpeg_quality, no_jpeg=no_jpeg)
    phase1_val = resized_dir_name("val", resize_size, crop_size, jpeg_quality, no_jpeg=no_jpeg)
    phase2_train = jpeg_only_dir_name("train", jpeg_quality, no_jpeg=no_jpeg)
    phase2_val = jpeg_only_dir_name("val", jpeg_quality, no_jpeg=no_jpeg)

    with open_dict(cfg.datamodule):
        cfg.datamodule.train_dir = phase1_train
        cfg.datamodule.val_dir = phase1_val
        cfg.datamodule.phase2_train_dir = phase2_train
        cfg.datamodule.phase2_val_dir = phase2_val
        cfg.datamodule.switch_epoch = data_cfg.switch_epoch
        cfg.datamodule.skip_resize_crop = data_cfg.get("skip_resize_crop", True)
        if data_cfg.get("disable_mixup_cutmix", False):
            cfg.datamodule.cutmix_alpha = 0.0
            cfg.datamodule.mixup_alpha = 0.0

    log.info(
        f"Compressed data config: phase 1 train={phase1_train}, val={phase1_val}; "
        f"phase 2 train={phase2_train}, val={phase2_val}; "
        f"switch_epoch={data_cfg.switch_epoch}, "
        f"disable_mixup_cutmix={data_cfg.get('disable_mixup_cutmix', False)}"
    )


def prepare_compressed_datasets(cfg: DictConfig) -> None:
    """Check for existing compressed datasets and run compression when needed."""
    data_cfg = cfg.data
    data_dir = Path(cfg.datamodule.data_path)
    resize_size = data_cfg.resize_size
    crop_size = data_cfg.crop_size
    jpeg_quality = data_cfg.jpeg_quality
    overwrite = data_cfg.get("overwrite", False)
    no_jpeg = data_cfg.get("no_jpeg", False)

    resized_train = resized_dir_name("train", resize_size, crop_size, jpeg_quality, no_jpeg=no_jpeg)
    resized_val = resized_dir_name("val", resize_size, crop_size, jpeg_quality, no_jpeg=no_jpeg)
    jpeg_train = jpeg_only_dir_name("train", jpeg_quality, no_jpeg=no_jpeg)
    jpeg_val = jpeg_only_dir_name("val", jpeg_quality, no_jpeg=no_jpeg)

    format_name = "PNG" if no_jpeg else "JPEG"
    quality_str = "PNG" if no_jpeg else f"quality={jpeg_quality}"
    log.info(
        f"Checking compressed datasets (resize={resize_size}, crop={crop_size}, "
        f"{quality_str}, overwrite={overwrite}): "
        f"resized=[{resized_train}, {resized_val}], "
        f"jpeg-only=[{jpeg_train}, {jpeg_val}]"
    )

    if not overwrite and compressed_datasets_exist(
        data_dir, resize_size, crop_size, jpeg_quality, no_jpeg=no_jpeg
    ):
        log.info(
            f"All compressed datasets already exist under {data_dir} and overwrite=false; "
            "skipping compression."
        )
        print(
            f"[compressed_data] Skipping compression: resized and {format_name}-only datasets "
            f"already exist in {data_dir} (set data.overwrite=true to reprocess)."
        )
        return

    if overwrite:
        log.info("overwrite=true: reprocessing all compressed datasets.")
        print("[compressed_data] overwrite=true: running compression (reprocessing all images).")
    else:
        log.info("One or more compressed datasets missing; running compression.")
        print("[compressed_data] Running compression to create missing datasets.")

    compress_splits(
        data_dir=data_dir,
        resize_size=resize_size,
        crop_size=crop_size,
        jpeg_quality=jpeg_quality,
        also_jpeg_only=True,
        overwrite=overwrite,
        no_jpeg=no_jpeg,
    )

    log.info("Compressed dataset preparation finished.")
    print("[compressed_data] Compression finished.")
