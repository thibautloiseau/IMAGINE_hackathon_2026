import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torchvision.transforms.v2 as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader


class IndexedDataset(Dataset):
    """Wraps a dataset so that each item additionally carries its integer index.

    Turns the usual `(image, target)` items into `(image, target, index)`. The index identifies
    the sample in the underlying dataset and is what the `LossBasedDataPruning` callback uses to
    track per-sample losses across epochs.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        return (*sample, index)


class UnlabeledImageFolder:
    # From https://github.com/pytorch/vision/issues/9050

    def __init__(self, root_dir, patterns=None, transform=None):
        self.root = Path(root_dir)
        self.images = []
        if patterns is None:
            patterns = [f"**/*{ext}" for ext in IMG_EXTENSIONS]
        for pattern in patterns:
            self.images.extend(self.root.glob(pattern, case_sensitive=False))
        self.images = sorted(self.images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = default_loader(self.images[i])
        if self.transform:
            img = self.transform(img)
        return img


class ImageNetDataModule(LightningDataModule):
    """`LightningDataModule` for the ImageNet dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_path: str = "data/",
        train_dir: str = "train",
        val_dir: str = "val",
        test_dir: str = "test",
        eval_resize_size: int = 256,
        eval_crop_size: int = 224,
        train_crop_size: int = 224,
        interpolation: str = "bilinear",
        hflip_prob: float = 0.0,
        auto_augment_policy: str = None,
        ra_magnitude: int = None,
        augmix_severity: int = None,
        cutmix_alpha: float = 0.0,
        mixup_alpha: float = 0.0,
        random_erase_prob: float = 0.0,
        train_augment: bool = True,
        batch_size: int = 64,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
    ) -> None:
        """Initialize an `ImageNetDataModule`.

        :param data_path: The data directory path. Defaults to `"data/"`.
        :param train_dir: The training data directory name. Defaults to `"train"`.
        :param val_dir: The validation data directory name. Defaults to `"val"`.
        :param test_dir: The test data directory name. Defaults to `"test"`.
        :param eval_resize_size: The size to resize the shorter side of the image for evaluation. Defaults to `256`.
        :param eval_crop_size: The size to center crop the image for evaluation. Defaults to `224`.
        :param train_crop_size: The size to randomly crop the image for training. Defaults to `224`.
        :param interpolation: The interpolation method to use for resizing. Defaults to `'bilinear'`.
        :param hflip_prob: The probability of applying random horizontal flip during training. Defaults to `0.0`.
        :param auto_augment_policy: The auto-augment policy to use during training. Can be one of `"ra"`, `"ta_wide"`, `"augmix"`, or any policy supported by `torchvision.transforms.AutoAugmentPolicy`. Defaults to `None` (no auto-augmentation).
        :param ra_magnitude: The magnitude to use for RandAugment if `auto_augment_policy` is set to `"ra"`. Defaults to `None`.
        :param augmix_severity: The severity to use for AugMix if `auto_augment_policy` is set to `"augmix"`. Defaults to `None`.
        :param cutmix_alpha: The alpha value for CutMix augmentation. Defaults to `0.0` (no CutMix).
        :param mixup_alpha: The alpha value for MixUp augmentation. Defaults to `0.0` (no MixUp).
        :param random_erase_prob: The probability of applying random erasing during training. Defaults to `0.0`.
        :param train_augment: Whether to apply any training-time data augmentation. When `False`, all
            augmentations are disabled (random resized crop, horizontal flip, auto-augment, random
            erasing, MixUp and CutMix) and training uses the same deterministic resize + center-crop
            pipeline as evaluation. Defaults to `True`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param prefetch_factor: The number of batches to prefetch. Defaults to `2`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        interpolation_mode = T.InterpolationMode(interpolation)
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        train_transforms = []
        train_transforms.append(
            T.RandomResizedCrop(train_crop_size, interpolation=interpolation_mode)
        )
        if hflip_prob > 0:
            train_transforms.append(T.RandomHorizontalFlip(hflip_prob))

        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                train_transforms.append(
                    T.RandAugment(interpolation=interpolation_mode, magnitude=ra_magnitude)
                )
            elif auto_augment_policy == "ta_wide":
                train_transforms.append(T.TrivialAugmentWide(interpolation=interpolation_mode))
            elif auto_augment_policy == "augmix":
                train_transforms.append(
                    T.AugMix(interpolation=interpolation_mode, severity=augmix_severity)
                )
            else:
                aa_policy = T.AutoAugmentPolicy(auto_augment_policy)
                train_transforms.append(
                    T.AutoAugment(policy=aa_policy, interpolation=interpolation_mode)
                )

        train_transforms.extend(
            [
                T.PILToTensor(),
                T.ToDtype(torch.float, scale=True),
                T.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )
        if random_erase_prob > 0:
            train_transforms.append(T.RandomErasing(p=random_erase_prob))
        train_transforms.append(T.ToPureTensor())
        self.train_transforms = T.Compose(train_transforms)

        self.eval_transforms = T.Compose(
            [
                T.Resize(eval_resize_size, interpolation=interpolation_mode),
                T.CenterCrop(eval_crop_size),
                T.PILToTensor(),
                T.ToDtype(torch.float, scale=True),
                T.Normalize(mean=imagenet_mean, std=imagenet_std),
                T.ToPureTensor(),
            ]
        )

        # Disable all training-time augmentation: use the deterministic eval pipeline for training
        # (no random resized crop / flip / auto-augment / random erasing) and no MixUp/CutMix.
        if not train_augment:
            self.train_transforms = self.eval_transforms

        if train_augment and (cutmix_alpha or mixup_alpha):
            self.mixup_cutmix = self._get_mixup_cutmix(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
            )
        else:
            self.mixup_cutmix = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        # Loss-based data pruning state. Disabled by default; enabled by the
        # `LossBasedDataPruning` callback via `enable_pruning()`. When enabled the training set is
        # wrapped in an `IndexedDataset` and sampled through a `SubsetRandomSampler` restricted to
        # the currently active sample indices, which the callback updates every epoch.
        self.prune: bool = False
        self.train_sampler: Optional[SubsetRandomSampler] = None
        self.active_indices: Optional[List[int]] = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of ImageNet-1k classes (1000).
        """
        return 1000

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        if stage in ("test", "predict") or stage is None:
            if not self.data_test:
                self.data_test = UnlabeledImageFolder(
                    os.path.join(self.hparams.data_path, self.hparams.test_dir),
                    transform=self.eval_transforms,
                )
        if stage in ("fit", "validate") or stage is None:
            if not self.data_train:
                self.data_train = ImageFolder(
                    os.path.join(self.hparams.data_path, self.hparams.train_dir),
                    transform=self.train_transforms,
                )

            if not self.data_val:
                self.data_val = ImageFolder(
                    os.path.join(self.hparams.data_path, self.hparams.val_dir),
                    transform=self.eval_transforms,
                )

    def enable_pruning(self) -> None:
        """Switch the training set into loss-based pruning mode.

        Called by the `LossBasedDataPruning` callback during `setup`. Takes effect the next time
        `train_dataloader()` is requested (which happens after all `setup` hooks have run).
        """
        self.prune = True

    def set_active_indices(self, indices: Sequence[int]) -> None:
        """Restrict the next training epoch to the given training-set sample indices.

        Updates the sampler in place so the change is picked up on the next epoch without having to
        rebuild the dataloader.

        :param indices: The dataset indices of the samples to keep for training.
        """
        self.active_indices = list(indices)
        if self.train_sampler is not None:
            self.train_sampler.indices = self.active_indices

    def _collate(self, batch: list) -> Any:
        """Collate a batch, applying MixUp/CutMix and (in pruning mode) preserving sample indices.

        :param batch: A list of `(image, target)` items, or `(image, target, index)` items when
            pruning is enabled.
        :return: `(images, targets)`, or `(images, targets, indices)` when pruning is enabled.
        """
        if self.prune:
            indices = torch.as_tensor([item[2] for item in batch])
            x, y = default_collate([(item[0], item[1]) for item in batch])
            if self.mixup_cutmix is not None:
                x, y = self.mixup_cutmix(x, y)
            return x, y, indices

        x, y = default_collate(batch)
        if self.mixup_cutmix is not None:
            x, y = self.mixup_cutmix(x, y)
        return x, y

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        dataset: Dataset = self.data_train
        sampler: Optional[SubsetRandomSampler] = None
        shuffle = True

        if self.prune:
            # Wrap so each item carries its index, and sample only the currently active subset.
            dataset = IndexedDataset(self.data_train)
            if self.active_indices is None:
                self.active_indices = list(range(len(self.data_train)))
            self.train_sampler = SubsetRandomSampler(self.active_indices)
            sampler = self.train_sampler
            shuffle = False

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn=self._collate,
            sampler=sampler,
            shuffle=shuffle,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader.

        :return: The predict dataloader.
        """
        return self.test_dataloader()

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    def _get_mixup_cutmix(self, mixup_alpha, cutmix_alpha):
        mixup_cutmix = []
        if mixup_alpha > 0:
            mixup_cutmix.append(T.MixUp(alpha=mixup_alpha, num_classes=self.num_classes))
        if cutmix_alpha > 0:
            mixup_cutmix.append(T.CutMix(alpha=cutmix_alpha, num_classes=self.num_classes))
        if not mixup_cutmix:
            return None

        return T.RandomChoice(mixup_cutmix)


if __name__ == "__main__":
    _ = ImageNetDataModule()
