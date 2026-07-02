import math
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class ImageNetModule(LightningModule):
    """`LightningModule` for ImageNet classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        compile: bool,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        main_scheduler: torch.optim.lr_scheduler,
        warmup_scheduler: torch.optim.lr_scheduler = None,
    ) -> None:
        """Initialize an `ImageNetModule`.

        :param net: The model to train.
        :param compile: Whether to use `torch.compile` on the model for training.
        :param warmup_epochs: The number of warmup epochs to use for training. If 0, no warmup scheduler will be used.
        :param main_scheduler: The main learning rate scheduler to use for training.
        :param warmup_scheduler: The learning rate scheduler to use for warmup.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True, ignore=["net"])

        self.net = net

        # loss function. `reduction="none"` keeps a per-sample loss so that
        # loss-based data pruning (see `LossBasedDataPruning` callback) can track
        # how well the model does on each individual training example.
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc1 = Accuracy(task="multiclass", num_classes=1000)
        self.train_acc5 = Accuracy(task="multiclass", num_classes=1000, top_k=5)
        self.val_acc1 = Accuracy(task="multiclass", num_classes=1000)
        self.val_acc5 = Accuracy(task="multiclass", num_classes=1000, top_k=5)
        self.test_acc1 = Accuracy(task="multiclass", num_classes=1000)
        self.test_acc5 = Accuracy(task="multiclass", num_classes=1000, top_k=5)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc1.reset()
        self.val_acc5.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels. When loss-based data pruning is enabled the batch additionally carries the
            dataset indices of each sample, i.e. `(images, targets, indices)`.

        :return: A tuple containing (in order):
            - A tensor of per-sample losses (shape `[batch_size]`).
            - A tensor of logits.
            - A tensor of target labels.
            - A tensor of dataset sample indices, or `None` when pruning is disabled.
        """
        if len(batch) == 3:
            x, y, idx = batch
        else:
            x, y = batch
            idx = None
        logits = self.forward(x)
        sample_loss = self.criterion(logits, y)
        if y.dim() > 1:
            y = y.argmax(dim=1)
        return sample_loss, logits, y.long(), idx

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets, or a dict additionally
            carrying the per-sample losses and dataset indices when pruning is enabled.
        """
        sample_loss, logits, targets, idx = self.model_step(batch)
        loss = sample_loss.mean()

        # update and log metrics
        self.train_loss(loss)
        self.train_acc1(logits, targets)
        self.train_acc5(logits, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc1", self.train_acc1, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc5", self.train_acc5, on_step=True, on_epoch=True, prog_bar=True)

        # When pruning is enabled, expose the per-sample losses and their dataset indices so the
        # `LossBasedDataPruning` callback can decide which samples to drop. Otherwise return the
        # scalar loss as usual. (Backpropagation always uses `loss`.)
        if idx is not None:
            return {
                "loss": loss,
                "sample_loss": sample_loss.detach(),
                "sample_top5_margin": self._top5_margin(logits, targets),
                "sample_idx": idx,
            }
        return loss

    @staticmethod
    def _top5_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-sample top-5 confidence signal for confidence-based data pruning.

        Measures how much probability mass the model concentrates on its five most likely classes
        relative to all the others: ``margin = sum(top-5 probs) - sum(remaining probs)``, which is
        positive exactly when the top-5 set holds more than half of the total probability mass.
        Samples whose true label is *not* inside the predicted top-5 are forced to the minimum value
        (``-1``) so they are never considered "mastered" — we don't want to prune an example the
        model still gets wrong at top-5. See `LossBasedDataPruning` (``criterion="top5_confidence"``).

        :param logits: Model logits, shape `[batch_size, num_classes]`.
        :param targets: Hard target labels, shape `[batch_size]`.
        :return: A tensor of top-5 margins in `[-1, 1]`, shape `[batch_size]`.
        """
        with torch.no_grad():
            probs = logits.softmax(dim=1)
            top5_vals, top5_idx = probs.topk(5, dim=1)
            top5_mass = top5_vals.sum(dim=1)
            margin = 2.0 * top5_mass - 1.0  # sum(top-5) - sum(rest); > 0 iff top-5 mass > 0.5
            in_top5 = (top5_idx == targets.unsqueeze(1)).any(dim=1)
            return torch.where(in_top5, margin, torch.full_like(margin, -1.0)).detach()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        sample_loss, logits, targets, _ = self.model_step(batch)
        loss = sample_loss.mean()

        # update and log metrics
        self.val_loss(loss)
        self.val_acc1(logits, targets)
        self.val_acc5(logits, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc1", self.val_acc1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc5", self.val_acc5, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        _, logits, targets, _ = self.model_step(batch)

        # update and log metrics
        self.test_acc1(logits, targets)
        self.test_acc5(logits, targets)
        self.log("test/acc1", self.test_acc1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc5", self.test_acc5, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Apply the per-step, epoch-anchored warmup + cosine learning rate for this step.

        The LR is a continuous function of *fractional* epoch progress
        ``p = current_epoch + batch_idx / num_training_batches``: a linear warmup from
        ``start_factor * lr`` up to the peak ``lr`` over ``warmup_epochs``, then a cosine anneal to
        ``eta_min`` reached exactly at ``max_epochs``. Measuring progress in epochs — with
        ``num_training_batches`` recomputed each epoch to match the pruned active subset
        (``reload_dataloaders_every_n_epochs=1``) — keeps the ramp smooth *within* every epoch (like
        the old step-based warmup) while still finishing precisely at ``max_epochs``, no matter how
        many steps data pruning removes.
        """
        if not hasattr(self, "_base_lrs"):
            return
        num_batches = self.trainer.num_training_batches
        if not num_batches or num_batches == float("inf"):
            return

        progress = self.trainer.current_epoch + batch_idx / num_batches
        warmup_epochs = self.hparams.warmup_epochs
        max_epochs = self.trainer.max_epochs

        if warmup_epochs > 0 and progress < warmup_epochs:
            start_factor = self._warmup_start_factor
            factor = start_factor + (1.0 - start_factor) * (progress / warmup_epochs)
            factors = [factor] * len(self._base_lrs)
        else:
            span = max(max_epochs - warmup_epochs, 1e-8)
            cos_progress = min(max((progress - warmup_epochs) / span, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * cos_progress))
            factors = [ratio + (1.0 - ratio) * cosine for ratio in self._eta_min_ratios]

        for group, base, factor in zip(self.trainer.optimizers[0].param_groups, self._base_lrs, factors):
            group["lr"] = base * factor
        self.log("train/lr", self._base_lrs[0] * factors[0], on_step=True, on_epoch=False)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the optimizer.

        The learning-rate schedule is deliberately *not* returned as a Lightning scheduler; it is
        applied per step in `on_train_batch_start`, driven by fractional-epoch progress. That yields
        a smooth within-epoch warmup (like a step-based schedule) while staying anchored to epochs,
        so it remains correct under dynamic data pruning, which changes the number of steps per
        epoch. The `warmup_scheduler` / `main_scheduler` configs are reused only for their shape
        parameters (`start_factor`, `eta_min`).

        :return: A dict containing the configured optimizer.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        # Peak LR per param group (the schedule scales these) and the schedule's shape parameters.
        self._base_lrs = [group["lr"] for group in optimizer.param_groups]
        warmup = self.hparams.warmup_scheduler
        main = self.hparams.main_scheduler
        self._warmup_start_factor = (
            float(warmup.keywords.get("start_factor", 1.0)) if warmup is not None else 1.0
        )
        eta_min = float(main.keywords.get("eta_min", 0.0)) if main is not None else 0.0
        self._eta_min_ratios = [eta_min / base if base else 0.0 for base in self._base_lrs]
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = ImageNetModule(None, None, None, None)
