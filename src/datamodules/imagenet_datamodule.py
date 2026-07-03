import math
import os
import tarfile
from functools import partial
from glob import glob
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torchvision.transforms.v2 as T
import webdataset
import webdataset as wds
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader


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
        train_tar: Optional[str] = None,
        val_tar: Optional[str] = None,
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
        batch_size: int = 64,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        wds=False,
        wds_buffer_size=1000,
        class_label_map: Optional[Dict[str, int]] = None,
    ) -> None:
        """Initialize an `ImageNetDataModule`.

        :param data_path: The data directory path. Defaults to `"data/"`.
        :param train_dir: The training data directory name. Defaults to `"train"`.
        :param val_dir: The validation data directory name. Defaults to `"val"`.
        :param test_dir: The test data directory name. Defaults to `"test"`.
        :param train_tar: Path to a training webdataset tar file. Defaults to `None`.
        :param val_tar: Path to a validation webdataset tar file. Defaults to `None`.
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
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param prefetch_factor: The number of batches to prefetch. Defaults to `2`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self._interpolation_mode = T.InterpolationMode(interpolation)
        self._imagenet_mean = (0.485, 0.456, 0.406)
        self._imagenet_std = (0.229, 0.224, 0.225)
        self._train_crop_size = train_crop_size
        self.train_transforms = self._build_train_transforms(train_crop_size)

        self.eval_transforms = T.Compose(
            [
                T.Resize(eval_resize_size, interpolation=self._interpolation_mode),
                T.CenterCrop(eval_crop_size),
                T.PILToTensor(),
                T.ToDtype(torch.float, scale=True),
                T.Normalize(mean=self._imagenet_mean, std=self._imagenet_std),
                T.ToPureTensor(),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.wds = wds
        if self.wds:
            # Expand globs so a shard pattern (e.g. train_jpeg50-*.tar) becomes the
            # list of shards; fall back to the literal path for a single tar / remote url.
            self.train_url = sorted(glob(train_tar)) or [train_tar] if train_tar else sorted(glob(f"{train_dir}/*.tar"))
            self.val_url = sorted(glob(val_tar)) or [val_tar] if val_tar else sorted(glob(f"{val_dir}/*.tar"))
            # If train and val share the same shards, check if keys have train/val prefix
            same_urls = self.train_url and self.val_url and self.train_url == self.val_url
            self._split_by_key = False
            if same_urls:
                ds = webdataset.WebDataset([self.train_url[0]], shardshuffle=False)
                try:
                    first_key = next(iter(ds)).get("__key__", "")
                except StopIteration:
                    first_key = ""
                if first_key.startswith(("train/", "val/")):
                    self._split_by_key = True
                    self.val_url = self.train_url  # shared shards, filtered by prefix
        self.buffer_size = wds_buffer_size
        self._wds_train_count = None
        self._wds_val_count = None

        # Build class_label_map automatically from the train directory if not provided
        if class_label_map is not None:
            self.class_label_map = class_label_map
        elif self.wds:
            train_path = os.path.join(data_path, train_dir)
            if Path(train_path).exists() and any(Path(train_path).iterdir()):
                class_dirs = sorted(
                    [d.name for d in Path(train_path).iterdir() if d.is_dir()]
                )
                self.class_label_map = {c: i for i, c in enumerate(class_dirs)}
            else:
                # Fallback: extract class names from webdataset keys
                class_names = set()
                for url in self.train_url[:3]:  # sample a few shards
                    ds = webdataset.WebDataset([url], shardshuffle=False)
                    for sample in ds:
                        key = sample.get("__key__", "")
                        cls = Path(key).parent.name
                        if cls:
                            class_names.add(cls)
                        if len(class_names) >= 1000:
                            break
                    if len(class_names) >= 1000:
                        break
                if not class_names:
                    raise FileNotFoundError(
                        f"Could not determine class names from train_dir ({train_path}) "
                        f"or webdataset shards ({self.train_url})."
                    )
                self.class_label_map = {c: i for i, c in enumerate(sorted(class_names))}
        else:
            self.class_label_map = None

        # Set up the collate function.
        # For webdataset the pipeline already yields batched (image, label) tuples,
        # so the DataLoader collate is a simple pass-through.
        if self.wds:
            self.collate_fn = lambda x: x
        elif cutmix_alpha or mixup_alpha:
            mixup_cutmix = self._get_mixup_cutmix(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
            )
            self.collate_fn = lambda batch: mixup_cutmix(*default_collate(batch))
        else:
            self.collate_fn = default_collate

    def _build_train_transforms(self, train_crop_size: int) -> T.Compose:
        train_transforms = [
            T.RandomResizedCrop(
                train_crop_size, interpolation=self._interpolation_mode
            ),
        ]
        if self.hparams.hflip_prob > 0:
            train_transforms.append(T.RandomHorizontalFlip(self.hparams.hflip_prob))

        auto_augment_policy = self.hparams.auto_augment_policy
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                train_transforms.append(
                    T.RandAugment(
                        interpolation=self._interpolation_mode,
                        magnitude=self.hparams.ra_magnitude,
                    )
                )
            elif auto_augment_policy == "ta_wide":
                train_transforms.append(
                    T.TrivialAugmentWide(interpolation=self._interpolation_mode)
                )
            elif auto_augment_policy == "augmix":
                train_transforms.append(
                    T.AugMix(
                        interpolation=self._interpolation_mode,
                        severity=self.hparams.augmix_severity,
                    )
                )
            else:
                aa_policy = T.AutoAugmentPolicy(auto_augment_policy)
                train_transforms.append(
                    T.AutoAugment(
                        policy=aa_policy, interpolation=self._interpolation_mode
                    )
                )

        train_transforms.extend(
            [
                T.PILToTensor(),
                T.ToDtype(torch.float, scale=True),
                T.Normalize(mean=self._imagenet_mean, std=self._imagenet_std),
            ]
        )
        if self.hparams.random_erase_prob > 0:
            train_transforms.append(T.RandomErasing(p=self.hparams.random_erase_prob))
        train_transforms.append(T.ToPureTensor())
        return T.Compose(train_transforms)

    def set_train_crop_size(self, crop_size: int) -> None:
        """Update training crop size and swap transforms on the train dataset."""
        self._train_crop_size = crop_size
        self.train_transforms = self._build_train_transforms(crop_size)
        if self.data_train is not None:
            self.data_train.transform = self.train_transforms

    def set_batch_size(self, batch_size: int) -> None:
        """Update per-device training batch size."""
        self.batch_size_per_device = batch_size

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

    def _count_wds_samples(self, urls, prefix=None):
        """Count total samples across webdataset shards (fast: samples one shard, extrapolates).

        If *prefix* is given (e.g. ``"train/"``), only counts entries whose tar name
        starts with that prefix.
        """
        count_per_shard = 0
        with tarfile.open(urls[0], 'r|*') as tar:
            for member in tar:
                if member.name.endswith(('.jpeg', '.jpg', '.JPEG', '.JPG')):
                    if prefix is None or member.name.startswith(prefix):
                        count_per_shard += 1
        return count_per_shard * len(urls)

    def _batches_per_epoch(self, train: bool) -> int:
        """Number of batches in one epoch, derived from the ImageFolder counts.

        Used both to bound the (length-less) webdataset train epoch and to give the
        WebLoaders a nominal ``__len__`` so Lightning's progress bar shows a total.
        """
        if self.wds:
            count = self._wds_train_count if train else self._wds_val_count
            return math.ceil(count / self.batch_size_per_device)
        data = self.data_train if train else self.data_val
        return math.ceil(len(data) / self.batch_size_per_device)

    def make_dataset(self, train: bool = False) -> Dataset:
        """Build a webdataset pipeline that decodes, transforms, and batches samples.

        Uses ``.decode("pil")`` to get PIL Images directly, then applies the
        train or eval transform.  Shuffling is only applied for the training split.
        """
        urls = self.train_url if train else self.val_url
        prefix = "train/" if train else "val/"
        dataset = wds.WebDataset(urls, shardshuffle=len(urls) if train else False)
        if self._split_by_key:
            dataset = dataset.select(lambda s: s.get("__key__", "").startswith(prefix))
        if train:
            dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.decode("pil")
        dataset = dataset.map(partial(self.wds_transform, train=train))
        # Val keeps the last partial batch so every sample is evaluated; train drops it.
        dataset = dataset.batched(self.batch_size_per_device, partial=not train)
        return dataset

    def wds_transform(self, sample, train: bool = False):
        img = sample["jpeg"].convert("RGB")
        class_name = Path(sample["__key__"]).parent.name
        label = torch.tensor(self.class_label_map[class_name], dtype=torch.long)
        return self.train_transforms(img) if train else self.eval_transforms(img), label

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        if not self.data_test:
            test_path = os.path.join(self.hparams.data_path, self.hparams.test_dir)
            if stage == "predict":
                self.data_test = UnlabeledImageFolder(
                    test_path,
                    transform=self.eval_transforms,
                )
            elif stage == "test":
                self.data_test = ImageFolder(
                    test_path,
                    transform=self.eval_transforms,
                )
        if stage in ("fit", "validate") or stage is None:
            if self.wds:
                if self._wds_train_count is None:
                    self._wds_train_count = self._count_wds_samples(
                        self.train_url,
                        prefix="train/" if self._split_by_key else None,
                    )
                    self._wds_val_count = self._count_wds_samples(
                        self.val_url,
                        prefix="val/" if self._split_by_key else None,
                    )
            else:
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

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """

        if self.wds:
            dataset = self.make_dataset(train=True)
            nbatches = self._batches_per_epoch(train=True)
            # Keep workers alive across epochs (avoids respawn + shuffle-buffer refill
            # stalls at every epoch boundary) and deepen the prefetch queue so the GPU
            # doesn't drain it mid-epoch. Both are only valid with num_workers > 0.
            worker_kwargs = (
                {
                    "prefetch_factor": self.hparams.prefetch_factor,
                    "persistent_workers": True,
                }
                if self.hparams.num_workers > 0
                else {}
            )
            loader = wds.WebLoader(
                dataset,
                batch_size=None,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.collate_fn,
                **worker_kwargs,
            )
            # Bound the stream to one nominal pass across all workers so the epoch
            # ends, validation fires, and the LR schedule aligns with the T_max
            # computed in train.py. For shard-splitting this truncates/repeats only
            # by the few batches lost to per-worker partial drops. with_length feeds
            # the progress bar.
            return loader.with_epoch(nbatches).with_length(nbatches)
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        if self.wds:
            dataset = self.make_dataset(train=False)
            # Val is small — keep few workers, no persistence, to avoid OOM from
            # 32 buffered prefetch queues (train + val workers alive simultaneously)
            val_workers = min(self.hparams.num_workers, 4)
            loader = wds.WebLoader(
                dataset,
                batch_size=None,
                num_workers=val_workers,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.collate_fn,
            )
            return loader.with_length(self._batches_per_epoch(train=False))
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
        if self.wds:
            dataset = self.make_dataset(train=False)
            return wds.WebLoader(
                dataset,
                batch_size=None,
                num_workers=0,
                pin_memory=self.hparams.pin_memory,
                collate_fn=self.collate_fn,
            )
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
            mixup_cutmix.append(
                T.MixUp(alpha=mixup_alpha, num_classes=self.num_classes)
            )
        if cutmix_alpha > 0:
            mixup_cutmix.append(
                T.CutMix(alpha=cutmix_alpha, num_classes=self.num_classes)
            )
        if not mixup_cutmix:
            return None

        return T.RandomChoice(mixup_cutmix)


if __name__ == "__main__":
    _ = ImageNetDataModule()
