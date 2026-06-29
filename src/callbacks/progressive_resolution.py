from typing import Literal, Optional

from lightning import Callback
from lightning.pytorch.utilities import rank_zero_only


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
    ) -> None:
        """Initialize the callback.

        :param mode: `"linear"` increases crop size in steps; `"direct"` jumps once to full size.
        :param full_crop_size: Final crop size. Defaults to `224`.
        :param step_size: Pixels added per dimension at each linear stage. Defaults to `16`.
        :param epochs_per_stage: Epochs per resolution in linear mode. Defaults to `1`.
        :param switch_epoch: First epoch (0-based) at full resolution in direct mode.
        :param start_crop_size: Initial crop size. If `None`, uses the datamodule's initial
            `train_crop_size` at epoch 0.
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

        self.mode = mode
        self.full_crop_size = full_crop_size
        self.step_size = step_size
        self.epochs_per_stage = epochs_per_stage
        self.switch_epoch = switch_epoch
        self.start_crop_size = start_crop_size
        self._schedule: Optional[list[int]] = None
        self._last_logged_crop_size: Optional[int] = None

    def _build_schedule(self, start_crop_size: int) -> list[int]:
        sizes = list(range(start_crop_size, self.full_crop_size + 1, self.step_size))
        if not sizes:
            return [self.full_crop_size]
        if sizes[-1] != self.full_crop_size:
            sizes.append(self.full_crop_size)
        return sizes

    def _crop_size_for_epoch(self, epoch: int, start_crop_size: int) -> int:
        if self.mode == "direct":
            if epoch < self.switch_epoch:
                return start_crop_size
            return self.full_crop_size

        if self._schedule is None:
            self._schedule = self._build_schedule(start_crop_size)
        stage = min(epoch // self.epochs_per_stage, len(self._schedule) - 1)
        return self._schedule[stage]

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        datamodule = trainer.datamodule
        start_crop_size = self.start_crop_size
        if start_crop_size is None:
            start_crop_size = datamodule.hparams.train_crop_size

        crop_size = self._crop_size_for_epoch(trainer.current_epoch, start_crop_size)
        datamodule.set_train_crop_size(crop_size)

        if self._last_logged_crop_size != crop_size:
            self._log_crop_size(pl_module, crop_size)
            self._last_logged_crop_size = crop_size

    @rank_zero_only
    def _log_crop_size(self, pl_module, crop_size: int) -> None:
        pl_module.log(
            "train/crop_size",
            float(crop_size),
            on_step=False,
            on_epoch=True,
        )
