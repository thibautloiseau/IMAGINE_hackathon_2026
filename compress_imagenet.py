#!/usr/bin/env python3
"""Resize, center-crop, and compress an ImageNet-style dataset.

Applies the same resize + center crop used for validation in
``ImageNetDataModule`` (shorter side resized, then center crop), then writes
images under ``data/`` with a name derived from the resize/crop sizes and
format, e.g. ``data/train_rs256_cc224_q75/`` (JPEG) or
``data/train_rs256_cc224_png/`` (PNG with ``--no-jpeg``).

With ``--also-jpeg-only``, each source image is read once and two outputs are
written: the resized/cropped dataset above and a recompression of the
original resolution in the same format, e.g. ``data/train_q75/``.

Example:
    uv run compress_imagenet.py --resize-size 256 --crop-size 224 --jpeg-quality 75
    uv run compress_imagenet.py --resize-size 128 --crop-size 96 --jpeg-quality 50 --also-jpeg-only
    uv run compress_imagenet.py --resize-size 128 --crop-size 96 --no-jpeg
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

Image.MAX_IMAGE_PIXELS = None

_INTERPOLATION = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}


@dataclass(frozen=True)
class ImageTask:
    src: Path
    resized_dst: Path
    jpeg_only_dst: Path | None
    resize_size: int
    crop_size: int
    interpolation: InterpolationMode
    jpeg_quality: int
    overwrite: bool
    no_jpeg: bool


def resized_dir_name(
    split: str,
    resize_size: int,
    crop_size: int,
    jpeg_quality: int,
    output_suffix: str | None = None,
    no_jpeg: bool = False,
) -> str:
    if output_suffix:
        return f"{split}_{output_suffix}"
    if no_jpeg:
        return f"{split}_rs{resize_size}_cc{crop_size}_png"
    return f"{split}_rs{resize_size}_cc{crop_size}_q{jpeg_quality}"


def jpeg_only_dir_name(
    split: str,
    jpeg_quality: int,
    output_suffix: str | None = None,
    no_jpeg: bool = False,
) -> str:
    if output_suffix:
        suffix = "png" if no_jpeg else "jpeg"
        return f"{split}_{output_suffix}_{suffix}"
    if no_jpeg:
        return f"{split}_png"
    return f"{split}_q{jpeg_quality}"


def _split_has_images(split_dir: Path) -> bool:
    if not split_dir.is_dir():
        return False
    for ext in IMG_EXTENSIONS:
        if any(split_dir.rglob(f"*{ext}")):
            return True
        if any(split_dir.rglob(f"*{ext.upper()}")):
            return True
    return False


def compressed_datasets_exist(
    data_dir: Path,
    resize_size: int,
    crop_size: int,
    jpeg_quality: int,
    splits: tuple[str, ...] = ("train", "val"),
    output_suffix: str | None = None,
    no_jpeg: bool = False,
) -> bool:
    """Return True if all resized+compressed and JPEG-only split dirs exist with images."""
    data_dir = data_dir.resolve()
    for split in splits:
        resized = data_dir / resized_dir_name(
            split, resize_size, crop_size, jpeg_quality, output_suffix, no_jpeg=no_jpeg
        )
        jpeg_only = data_dir / jpeg_only_dir_name(split, jpeg_quality, output_suffix, no_jpeg=no_jpeg)
        if not _split_has_images(resized) or not _split_has_images(jpeg_only):
            return False
    return True


def _collect_tasks(
    src_root: Path,
    resized_root: Path,
    jpeg_only_root: Path | None,
    resize_size: int,
    crop_size: int,
    interpolation: InterpolationMode,
    jpeg_quality: int,
    overwrite: bool,
    no_jpeg: bool,
) -> list[ImageTask]:
    tasks: list[ImageTask] = []
    ext = ".png" if no_jpeg else ".JPEG"
    for src_path in sorted(src_root.rglob("*")):
        if not src_path.is_file():
            continue
        if src_path.suffix.lower() not in {ext.lower() for ext in IMG_EXTENSIONS}:
            continue

        rel_path = src_path.relative_to(src_root)
        jpeg_only_dst = None
        if jpeg_only_root is not None:
            jpeg_only_dst = jpeg_only_root / rel_path.with_suffix(ext)

        tasks.append(
            ImageTask(
                src=src_path,
                resized_dst=resized_root / rel_path.with_suffix(ext),
                jpeg_only_dst=jpeg_only_dst,
                resize_size=resize_size,
                crop_size=crop_size,
                interpolation=interpolation,
                jpeg_quality=jpeg_quality,
                overwrite=overwrite,
                no_jpeg=no_jpeg,
            )
        )
    return tasks


def _needs_processing(dst: Path, overwrite: bool) -> bool:
    return overwrite or not dst.exists()


def _save_image(img: Image.Image, dst: Path, jpeg_quality: int, no_jpeg: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if no_jpeg:
        img.save(dst, format="PNG", optimize=True)
    else:
        img.save(dst, format="JPEG", quality=jpeg_quality, optimize=True)


def _process_image(task: ImageTask) -> str:
    need_resized = _needs_processing(task.resized_dst, task.overwrite)
    need_jpeg_only = (
        task.jpeg_only_dst is not None
        and _needs_processing(task.jpeg_only_dst, task.overwrite)
    )
    if not need_resized and not need_jpeg_only:
        return "skipped"

    with Image.open(task.src) as img:
        rgb = img.convert("RGB") if img.mode != "RGB" else img

        if need_jpeg_only:
            _save_image(rgb, task.jpeg_only_dst, task.jpeg_quality, no_jpeg=task.no_jpeg)

        if need_resized:
            resized = F.resize(rgb, task.resize_size, interpolation=task.interpolation)
            cropped = F.center_crop(resized, task.crop_size)
            _save_image(cropped, task.resized_dst, task.jpeg_quality, no_jpeg=task.no_jpeg)

    return "processed"


def _process_split(
    data_dir: Path,
    split: str,
    resize_size: int,
    crop_size: int,
    interpolation: InterpolationMode,
    jpeg_quality: int,
    output_suffix: str | None,
    also_jpeg_only: bool,
    overwrite: bool,
    num_workers: int,
    console: Console,
    no_jpeg: bool,
) -> None:
    src_root = data_dir / split
    if not src_root.is_dir():
        raise FileNotFoundError(f"Split directory not found: {src_root}")

    resized_root = data_dir / resized_dir_name(
        split, resize_size, crop_size, jpeg_quality, output_suffix, no_jpeg=no_jpeg
    )
    jpeg_only_root = None
    if also_jpeg_only:
        jpeg_only_root = data_dir / jpeg_only_dir_name(split, jpeg_quality, output_suffix, no_jpeg=no_jpeg)

    tasks = _collect_tasks(
        src_root,
        resized_root,
        jpeg_only_root,
        resize_size,
        crop_size,
        interpolation,
        jpeg_quality,
        overwrite,
        no_jpeg,
    )
    if not tasks:
        console.print(f"[yellow]No images found in {src_root}[/yellow]")
        return

    console.print(f"[bold]{split}[/bold]: {len(tasks):,} images")
    console.print(f"  resized -> {resized_root}")
    if jpeg_only_root is not None:
        console.print(f"  jpeg-only -> {jpeg_only_root}")

    counts = {"processed": 0, "skipped": 0, "failed": 0}
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        job = progress.add_task(split, total=len(tasks))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_image, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    counts[result] += 1
                except Exception as exc:
                    counts["failed"] += 1
                    console.print(f"[red]Failed[/red] {task.src}: {exc}")
                progress.advance(job)

    console.print(
        f"  processed={counts['processed']:,}, "
        f"skipped={counts['skipped']:,}, "
        f"failed={counts['failed']:,}"
    )


def compress_splits(
    data_dir: Path,
    resize_size: int,
    crop_size: int,
    jpeg_quality: int,
    splits: tuple[str, ...] = ("train", "val"),
    also_jpeg_only: bool = True,
    overwrite: bool = False,
    interpolation: str = "bilinear",
    output_suffix: str | None = None,
    num_workers: int | None = None,
    no_jpeg: bool = False,
    console: Console | None = None,
) -> None:
    """Resize, crop, and compress dataset splits (optionally uncompressed copies too)."""
    if resize_size <= 0:
        raise ValueError("resize_size must be positive.")
    if crop_size <= 0:
        raise ValueError("crop_size must be positive.")
    if crop_size > resize_size:
        raise ValueError("crop_size must be <= resize_size.")
    if not no_jpeg and not 1 <= jpeg_quality <= 95:
        raise ValueError("jpeg_quality must be between 1 and 95.")

    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    console = console or Console()
    workers = num_workers or max(1, os.cpu_count() or 1)
    interpolation_mode = _INTERPOLATION[interpolation]

    format_name = "PNG" if no_jpeg else "JPEG"
    quality_info = f"{format_name}" if no_jpeg else f"{format_name} quality={jpeg_quality}"
    console.print(
        f"Resize={resize_size}, crop={crop_size}, "
        f"interpolation={interpolation}, {quality_info}, "
        f"also_jpeg_only={also_jpeg_only}, "
        f"workers={workers}, overwrite={overwrite}"
    )

    for split in splits:
        _process_split(
            data_dir=data_dir,
            split=split,
            resize_size=resize_size,
            crop_size=crop_size,
            interpolation=interpolation_mode,
            jpeg_quality=jpeg_quality,
            output_suffix=output_suffix,
            also_jpeg_only=also_jpeg_only,
            overwrite=overwrite,
            num_workers=workers,
            console=console,
            no_jpeg=no_jpeg,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize, center-crop, and JPEG-compress an ImageNet-style dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory containing train/ and val/ splits (default: data).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to process (default: train val).",
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        required=True,
        help="Resize the shorter side to this size, matching eval_resize_size in the datamodule.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        required=True,
        help="Center-crop to this square size, matching eval_crop_size in the datamodule.",
    )
    parser.add_argument(
        "--interpolation",
        choices=sorted(_INTERPOLATION),
        default="bilinear",
        help="Interpolation method for resizing (default: bilinear).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=75,
        help="JPEG compression quality from 1 (lowest) to 95 (highest). Ignored with --no-jpeg. (default: 75).",
    )
    parser.add_argument(
        "--no-jpeg",
        action="store_true",
        help="Save images as lossless PNG instead of JPEG. Disables JPEG compression.",
    )
    parser.add_argument(
        "--also-jpeg-only",
        action="store_true",
        help=(
            "Also write a JPEG-only recompression of each image at the original resolution, "
            "using the same quality. Both outputs are produced in a single pass per image."
        ),
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help=(
            "Optional suffix for output folders. Resized output goes to data/<split>_<suffix>/; "
            "with --also-jpeg-only, JPEG-only output goes to data/<split>_<suffix>_jpeg/."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: CPU count).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess images even when the destination file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    try:
        compress_splits(
            data_dir=args.data_dir,
            resize_size=args.resize_size,
            crop_size=args.crop_size,
            jpeg_quality=args.jpeg_quality,
            splits=tuple(args.splits),
            also_jpeg_only=args.also_jpeg_only,
            overwrite=args.overwrite,
            interpolation=args.interpolation,
            output_suffix=args.output_suffix,
            num_workers=args.num_workers,
            no_jpeg=args.no_jpeg,
            console=console,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
