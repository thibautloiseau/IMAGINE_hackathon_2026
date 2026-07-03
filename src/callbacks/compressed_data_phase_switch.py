import gc

import torch
from lightning import Callback, Trainer
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.combined_loader import _shutdown_workers_and_reset_iterator

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def _shutdown_dataloader(dataloader) -> None:
    """Shut down DataLoader worker processes."""
    if dataloader is None:
        return
    _shutdown_workers_and_reset_iterator(dataloader)


def _release_combined_loader(combined_loader) -> None:
    if combined_loader is None:
        return
    combined_loader.reset()
    for dataloader in list(combined_loader.flattened):
        _shutdown_dataloader(dataloader)


def _invalidate_dataloaders(trainer: Trainer) -> None:
    """Teardown Lightning loaders and drop phase-1 dataloader/dataset references."""
    fit_loop = trainer.fit_loop

    if fit_loop._data_fetcher is not None:
        if getattr(fit_loop._data_fetcher, "_combined_loader", None) is not None:
            _release_combined_loader(fit_loop._data_fetcher._combined_loader)
            fit_loop._data_fetcher._combined_loader = None
        fit_loop._data_fetcher.teardown()
        fit_loop._data_fetcher = None

    _release_combined_loader(fit_loop._combined_loader)
    fit_loop._combined_loader = None

    val_loop = fit_loop.epoch_loop.val_loop
    if val_loop._data_fetcher is not None:
        if getattr(val_loop._data_fetcher, "_combined_loader", None) is not None:
            _release_combined_loader(val_loop._data_fetcher._combined_loader)
            val_loop._data_fetcher._combined_loader = None
        val_loop._data_fetcher.teardown()
        val_loop._data_fetcher = None

    _release_combined_loader(val_loop._combined_loader)
    val_loop._combined_loader = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class CompressedDataPhaseSwitchCallback(Callback):
    """Switch from resized+compressed data to JPEG-only data at a configured epoch."""

    def __init__(self, switch_epoch: int) -> None:
        if switch_epoch < 0:
            raise ValueError(f"switch_epoch must be non-negative, got {switch_epoch}")
        self.switch_epoch = switch_epoch

    def _switch_to_phase_2(self, trainer: Trainer, completed_epoch: int) -> None:
        datamodule = trainer.datamodule
        if datamodule is None or datamodule.current_phase >= 2:
            return

        old_train = datamodule.hparams.train_dir
        old_val = datamodule.hparams.val_dir

        _invalidate_dataloaders(trainer)
        datamodule.switch_to_phase_2()
        self._log_switch(
            completed_epoch,
            old_train,
            old_val,
            datamodule,
        )

    def on_fit_start(self, trainer: Trainer, pl_module) -> None:
        if trainer.current_epoch >= self.switch_epoch:
            self._switch_to_phase_2(trainer, trainer.current_epoch)

    def on_train_epoch_end(self, trainer: Trainer, pl_module) -> None:
        if trainer.current_epoch + 1 == self.switch_epoch:
            self._switch_to_phase_2(trainer, trainer.current_epoch)

    @rank_zero_only
    def _log_switch(
        self,
        completed_epoch: int,
        old_train: str,
        old_val: str,
        datamodule,
    ) -> None:
        msg = (
            f"Switching to phase 2 (JPEG-only) after epoch {completed_epoch}: "
            f"train {old_train} -> {datamodule.hparams.train_dir}, "
            f"val {old_val} -> {datamodule.hparams.val_dir} "
            f"(online resize/crop enabled from epoch {completed_epoch + 1})"
        )
        log.info(msg)
        print(f"[compressed_data] {msg}")