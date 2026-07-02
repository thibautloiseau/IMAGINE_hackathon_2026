from typing import Any, Dict, List, Optional

import torch
from lightning import Callback, LightningModule, Trainer

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

# Sentinel "active again at epoch" value used to mark a sample as removed forever.
_FOREVER = 1_000_000_000


class LossBasedDataPruning(Callback):
    """Drop training samples the model has already mastered.

    For every example it sees, the model reports a per-sample "prune signal" (see
    `ImageNetModule.training_step`). This callback keeps an exponential moving average (EMA) of that
    signal per sample and, once a short warm-up has passed, removes every sample the model is
    consistently good at — so that training time is spent on the harder, still-informative samples.

    Two signals are supported via ``criterion``:

    * ``"loss"`` (default): the per-sample training loss (`sample_loss`). A sample is removed once
      its smoothed loss drops **below** `threshold`.
    * ``"top5_confidence"``: a per-sample top-5 confidence margin (`sample_top5_margin`), equal to
      `sum(top-5 probs) - sum(rest)` when the true label is inside the predicted top-5 and `-1`
      otherwise. A sample is removed once its smoothed margin rises **above** `threshold` (with the
      natural cutoff `threshold=0`, i.e. the top-5 holds more probability mass than everything else
      combined). Aimed at optimizing top-5 accuracy without dropping samples still wrong at top-5.

    Removal can be permanent or temporary:

    * ``reactivate_after=None`` removes an easy sample for the rest of training.
    * ``reactivate_after=N`` removes it for ``N`` epochs, after which it is shown again and
      re-evaluated (its loss may have drifted up as the model changed).

    A safety floor (`min_keep_fraction`) guarantees a minimum fraction of the dataset always stays
    active: if a round of pruning would drop below it, the hardest of the to-be-removed samples are
    kept instead.

    Notes / limitations:
        * Designed for single-device training. The per-sample bookkeeping and the
          `SubsetRandomSampler` it drives are not distributed-aware.
        * With MixUp/CutMix enabled the per-sample loss is measured on the *mixed* sample, so the
          pruning signal is an approximation. EMA smoothing absorbs most of that noise.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        criterion: str = "loss",
        ema_momentum: float = 0.9,
        warmup_epochs: int = 3,
        reactivate_after: Optional[int] = None,
        reactivate_prob: Optional[float] = None,
        min_keep_fraction: float = 0.1,
        verbose: bool = True,
    ) -> None:
        """Initialize a `LossBasedDataPruning` callback.

        :param threshold: Cutoff on the smoothed per-sample prune signal above/below which a sample
            counts as "learned" and is removed. With ``criterion="loss"`` a sample is removed once
            its EMA loss falls **below** this value (cross-entropy starts around `ln(num_classes)`,
            ~6.9 for ImageNet-1k, and easy, confidently-correct samples fall well below `1.0`). With
            ``criterion="top5_confidence"`` a sample is removed once its EMA top-5 margin rises
            **above** this value (`0.0` = the top-5 predictions hold over half of the probability
            mass). Defaults to `0.1`.
        :param criterion: Which per-sample signal drives pruning: ``"loss"`` (per-sample training
            loss) or ``"top5_confidence"`` (top-5 confidence margin, for optimizing top-5 accuracy).
            Defaults to ``"loss"``.
        :param ema_momentum: Momentum of the per-sample loss EMA, in `[0, 1)`. `0.0` uses only the
            most recent epoch's loss; higher values smooth more aggressively. Defaults to `0.9`.
        :param warmup_epochs: Number of epochs to train on the full dataset before any pruning
            happens, so the loss estimates can settle. Defaults to `3`.
        :param reactivate_after: Deterministic reintegration. If an int `N`, removed samples are
            skipped for exactly `N` epochs and then shown again and re-evaluated. If `None`, removed
            samples stay removed (unless `reactivate_prob` brings them back). Defaults to `None`.
        :param reactivate_prob: Stochastic reintegration. If set, at the end of every epoch each
            currently-removed sample independently rejoins the training set with this probability
            (e.g. `0.1` = 10% chance per epoch). Can be combined with `reactivate_after` (a sample
            returns as soon as either rule fires). Defaults to `None` (disabled).
        :param min_keep_fraction: Always keep at least this fraction of the dataset active. Defaults
            to `0.1`.
        :param verbose: Whether to log pruning statistics each epoch. Defaults to `True`.
        """
        super().__init__()
        if criterion not in ("loss", "top5_confidence"):
            raise ValueError(
                f"Unknown criterion {criterion!r}; expected 'loss' or 'top5_confidence'."
            )
        self.threshold = threshold
        self.criterion = criterion
        # Key of the per-sample signal to read from `training_step` outputs for this criterion.
        self._signal_key = "sample_loss" if criterion == "loss" else "sample_top5_margin"
        self.ema_momentum = ema_momentum
        self.warmup_epochs = warmup_epochs
        self.reactivate_after = reactivate_after
        self.reactivate_prob = reactivate_prob
        self.min_keep_fraction = min_keep_fraction
        self.verbose = verbose

        # State (allocated once the dataset size is known, in `on_train_start`).
        self.num_samples: Optional[int] = None
        self.sample_score: Optional[torch.Tensor] = None  # EMA of per-sample signal (NaN = unseen)
        self.seen: Optional[torch.Tensor] = None  # whether a sample has ever contributed to the EMA
        self.inactive_until: Optional[torch.Tensor] = None  # first epoch a sample is active again

    # ------------------------------------------------------------------ setup / state

    def _init_state(self, num_samples: int) -> None:
        self.num_samples = num_samples
        self.sample_score = torch.full((num_samples,), float("nan"))
        self.seen = torch.zeros(num_samples, dtype=torch.bool)
        self.inactive_until = torch.zeros(num_samples, dtype=torch.long)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if stage != "fit":
            return
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "enable_pruning"):
            raise RuntimeError(
                "LossBasedDataPruning requires a datamodule exposing `enable_pruning()` / "
                "`set_active_indices()` (e.g. ImageNetDataModule)."
            )
        datamodule.enable_pruning()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        num_samples = len(trainer.datamodule.data_train)
        # Allocate fresh state, unless a checkpoint already restored a matching one.
        if self.sample_score is None or self.num_samples != num_samples:
            self._init_state(num_samples)
        trainer.datamodule.set_active_indices(self._active_indices(trainer.current_epoch))

    # ------------------------------------------------------------------ per-batch tracking

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not isinstance(outputs, dict) or self._signal_key not in outputs:
            return

        signal = outputs[self._signal_key].detach().float().cpu()
        idx = outputs["sample_idx"].detach().cpu().long()

        prev = self.sample_score[idx]
        m = self.ema_momentum
        updated = torch.where(torch.isnan(prev), signal, m * prev + (1.0 - m) * signal)
        self.sample_score[idx] = updated
        self.seen[idx] = True

    # ------------------------------------------------------------------ per-epoch pruning

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch

        # Active set during the epoch that just finished (before this round's update).
        active_now_mask = self.inactive_until <= epoch

        # Train on the full dataset during warm-up so loss estimates can settle.
        if epoch + 1 >= self.warmup_epochs:
            self._update_removal(epoch)

        # Stochastic reintegration: each currently-removed sample independently rejoins the training
        # set with probability `reactivate_prob`. Done after filtering so a reintegrated sample is
        # actually trained next epoch before it can be re-evaluated (and re-filtered).
        if self.reactivate_prob:
            removed = self.inactive_until > (epoch + 1)
            draw = torch.rand(self.num_samples) < self.reactivate_prob
            self.inactive_until[removed & draw] = epoch + 1

        # Active set for the next epoch, and what changed relative to this epoch.
        active_next_mask = self.inactive_until <= (epoch + 1)
        active = torch.nonzero(active_next_mask, as_tuple=False).flatten().tolist()
        trainer.datamodule.set_active_indices(active)

        num_active = len(active)
        num_removed = self.num_samples - num_active
        newly_filtered = int((active_now_mask & ~active_next_mask).sum())
        reactivated = int((~active_now_mask & active_next_mask).sum())

        pl_module.log("prune/num_active", float(num_active), on_epoch=True, sync_dist=False)
        pl_module.log("prune/num_removed", float(num_removed), on_epoch=True, sync_dist=False)
        pl_module.log("prune/newly_filtered", float(newly_filtered), on_epoch=True, sync_dist=False)
        pl_module.log(
            "prune/active_fraction",
            num_active / max(self.num_samples, 1),
            on_epoch=True,
            sync_dist=False,
        )
        if self.verbose:
            log.info(
                f"[loss-pruning] epoch {epoch}: filtered {newly_filtered} example(s) this epoch "
                f"({reactivated} reactivated) -> {num_removed}/{self.num_samples} total filtered, "
                f"{num_active} active next epoch."
            )

    def _update_removal(self, epoch: int) -> None:
        """Remove newly-learned samples and apply the minimum-keep safety floor."""
        next_epoch = epoch + 1

        # Samples that were active this epoch, have a signal estimate, and cross the threshold. For
        # loss the "learned" side is below the threshold; for top-5 confidence it is above it.
        # (NaN comparisons are False, so unseen samples are excluded even before the `seen` mask.)
        currently_active = self.inactive_until <= epoch
        if self.criterion == "loss":
            crossed = self.sample_score < self.threshold
        else:  # top5_confidence
            crossed = self.sample_score > self.threshold
        learned = self.seen & currently_active & crossed

        if self.reactivate_after is None:
            self.inactive_until[learned] = _FOREVER
        else:
            self.inactive_until[learned] = next_epoch + self.reactivate_after

        self._enforce_min_keep(next_epoch)

    def _enforce_min_keep(self, next_epoch: int) -> None:
        """Reactivate the hardest removed samples if too few would remain active."""
        min_active = int(self.min_keep_fraction * self.num_samples)
        inactive_mask = self.inactive_until > next_epoch
        num_active = self.num_samples - int(inactive_mask.sum())
        if num_active >= min_active:
            return

        need = min_active - num_active
        removed_idx = torch.nonzero(inactive_mask, as_tuple=False).flatten()
        # Reactivate the least-learned of the removed samples: highest loss, or lowest top-5 margin.
        descending = self.criterion == "loss"
        order = torch.argsort(self.sample_score[removed_idx], descending=descending)
        to_reactivate = removed_idx[order[:need]]
        self.inactive_until[to_reactivate] = 0

    def _active_indices(self, epoch: int) -> List[int]:
        mask = self.inactive_until <= epoch
        return torch.nonzero(mask, as_tuple=False).flatten().tolist()

    # ------------------------------------------------------------------ checkpointing

    def state_dict(self) -> Dict[str, Any]:
        if self.sample_score is None:
            return {}
        return {
            "num_samples": self.num_samples,
            "sample_score": self.sample_score,
            "seen": self.seen,
            "inactive_until": self.inactive_until,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not state_dict:
            return
        self.num_samples = state_dict["num_samples"]
        # Fall back to the legacy "sample_loss" key for checkpoints written before `criterion`.
        self.sample_score = state_dict.get("sample_score", state_dict.get("sample_loss"))
        self.seen = state_dict["seen"]
        self.inactive_until = state_dict["inactive_until"]
