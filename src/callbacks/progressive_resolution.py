from typing import Literal, Optional

from lightning import Callback
from lightning.pytorch.utilities import rank_zero_only

from src.utils.progressive_resolution import (
    crop_size_for_epoch,
    token_matched_batch_size,
    token_matched_learning_rate,
)


class ProgressiveResolutionCallback(Callback):
    """Increase training crop size for progressive resolution training."""

    def __init__(
        self,
        mode: Literal["linear", "direct"] = "linear",
        full_crop_size: int = 224,
        step_size: int = 16,
        epochs_per_stage: int = 1,
        switch_epoch: Optional[int] = None,
        start_crop_size: Optional[int] = None,
        match_tokens: bool = False,
        reference_batch_size: int = 512,
        reference_lr: Optional[float] = None,
        patch_size: int = 16,
    ) -> None:
        """Initialize the callback.

        :param mode: `"linear"` increases crop size in steps; `"direct"` jumps once to full size.
        :param full_crop_size: Final crop size. Defaults to `224`.
        :param step_size: Pixels added per dimension at each linear stage. Defaults to `16`.
        :param epochs_per_stage: Epochs per resolution in linear mode. Defaults to `1`.
        :param switch_epoch: First epoch (0-based) at full resolution in direct mode.
        :param start_crop_size: Initial crop size. If `None`, uses the datamodule's initial
            `train_crop_size` at epoch 0.
        :param match_tokens: Scale batch size so patch tokens per step stay constant, and scale
            learning rate linearly with batch size. Defaults to `False`.
        :param reference_batch_size: Per-device batch size at `full_crop_size`. Defaults to `512`.
        :param reference_lr: Base LR at full resolution. If `None`, uses the module optimizer LR.
        :param patch_size: ViT patch size for token counting. Defaults to `16`.
        """
        if mode not in ("linear", "direct"):
            raise ValueError(f"mode must be 'linear' or 'direct', got {mode!r}")
        if mode == "direct" and switch_epoch is None:
            raise ValueError("switch_epoch is required when mode='direct'")
        if mode == "linear" and step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if epochs_per_stage <= 0:
            raise ValueError(f"epochs_per_stage must be positive, got {epochs_per_stage}")
        if start_crop_size is not None and start_crop_size > full_crop_size:
            raise ValueError(
                f"start_crop_size ({start_crop_size}) must be <= full_crop_size ({full_crop_size})"
            )
        if reference_batch_size <= 0:
            raise ValueError(f"reference_batch_size must be positive, got {reference_batch_size}")

        self.mode = mode
        self.full_crop_size = full_crop_size
        self.step_size = step_size
        self.epochs_per_stage = epochs_per_stage
        self.switch_epoch = switch_epoch
        self.start_crop_size = start_crop_size
        self.match_tokens = match_tokens
        self.reference_batch_size = reference_batch_size
        self.reference_lr = reference_lr
        self.patch_size = patch_size
        self._schedule: Optional[list[int]] = None
        self._last_state: Optional[tuple[int, int, float]] = None

    def _resolve_start_crop_size(self, datamodule) -> int:
        if self.start_crop_size is not None:
            return self.start_crop_size
        return datamodule.hparams.train_crop_size

    def _resolve_reference_lr(self, pl_module) -> float:
        if self.reference_lr is not None:
            return self.reference_lr
        optimizer_cfg = pl_module.hparams.optimizer
        if hasattr(optimizer_cfg, "keywords") and "lr" in optimizer_cfg.keywords:
            return float(optimizer_cfg.keywords["lr"])
        if hasattr(optimizer_cfg, "lr"):
            return float(optimizer_cfg.lr)
        raise ValueError("Could not resolve reference learning rate from module optimizer config")

    def _crop_size_for_epoch(self, epoch: int, start_crop_size: int) -> int:
        return crop_size_for_epoch(
            epoch,
            start_crop_size,
            self.full_crop_size,
            self.mode,
            self.step_size,
            self.epochs_per_stage,
            self.switch_epoch,
            self._schedule,
        )

    def _batch_size_for_crop(self, crop_size: int) -> int:
        return token_matched_batch_size(
            crop_size,
            self.full_crop_size,
            self.reference_batch_size,
        )

    def _reload_train_dataloader(self, trainer) -> None:
        trainer.fit_loop._combined_loader = None
        trainer.fit_loop.setup_data()

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        datamodule = trainer.datamodule
        start_crop_size = self._resolve_start_crop_size(datamodule)
        crop_size = self._crop_size_for_epoch(trainer.current_epoch, start_crop_size)
        batch_size = datamodule.batch_size_per_device
        learning_rate = None

        datamodule.set_train_crop_size(crop_size)

        if self.match_tokens:
            batch_size = self._batch_size_for_crop(crop_size)
            datamodule.set_batch_size(batch_size)
            reference_lr = self._resolve_reference_lr(pl_module)
            learning_rate = token_matched_learning_rate(
                crop_size,
                self.full_crop_size,
                reference_lr,
                self.reference_batch_size,
            )
            pl_module.set_base_learning_rate(learning_rate)

        state = (crop_size, batch_size, learning_rate or 0.0)
        if state != self._last_state:
            if self.match_tokens:
                self._reload_train_dataloader(trainer)
            self._log_training_state(pl_module, crop_size, batch_size, learning_rate)
            self._last_state = state

    @rank_zero_only
    def _log_training_state(
        self,
        pl_module,
        crop_size: int,
        batch_size: int,
        learning_rate: Optional[float],
    ) -> None:
        pl_module.log("train/crop_size", float(crop_size), on_step=False, on_epoch=True)
        if self.match_tokens:
            pl_module.log("train/batch_size", float(batch_size), on_step=False, on_epoch=True)
            if learning_rate is not None:
                pl_module.log("train/base_lr", learning_rate, on_step=False, on_epoch=True)
