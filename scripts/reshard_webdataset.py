"""Reshard a class-sorted webdataset tar into many globally-shuffled shards.

The source tar (e.g. ``train_jpeg50.tar``) stores samples grouped by class, so a
streaming shuffle buffer can never mix classes within a batch, and a single shard
can't be split across dataloader workers. This one-off script random-accesses the
source tar (preserving the original JPEG-50 bytes), shuffles sample order across
the whole dataset, and writes N shards suitable for ``shardshuffle=True`` +
multi-worker loading.

Usage:
    uv run scripts/reshard_webdataset.py \
        --src data/train_jpeg50.tar \
        --out-pattern data/shards/train_jpeg50-%06d.tar \
        --maxcount 10000
"""

import argparse
import random
import tarfile
from pathlib import Path

import webdataset as wds

IMG_EXTS = (".jpeg", ".jpg", ".png")


def sample_key(name: str) -> str:
    """``.../train/n03026506/n03026506_1316.JPEG`` -> ``train/n03026506/n03026506_1316``.

    Preserves the ``split/wnid/stem`` structure so the datamodule can filter
    train/val by key prefix and derive the label from ``Path(__key__).parent.name``.
    """
    p = Path(name)
    parts = p.parts
    return f"{parts[-3]}/{parts[-2]}/{p.stem}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="Source .tar (uncompressed)")
    ap.add_argument(
        "--out-pattern",
        required=True,
        help="printf-style shard pattern, e.g. data/shards/train_jpeg50-%%06d.tar",
    )
    ap.add_argument("--maxcount", type=int, default=10000, help="Max samples per shard")
    ap.add_argument(
        "--maxsize", type=float, default=1e9, help="Max bytes per shard (rolls first)"
    )
    ap.add_argument(
        "--prefix",
        default=None,
        help="Only include entries whose tar path starts with this prefix (e.g. 'train/' or 'val/')",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.out_pattern).parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(args.src, "r") as tar:
        members = [
            m
            for m in tar.getmembers()
            if m.isfile()
            and m.name.lower().endswith(IMG_EXTS)
            and (args.prefix is None or m.name.startswith(args.prefix))
        ]
        print(f"{len(members)} image samples in {args.src}")
        random.Random(args.seed).shuffle(members)

        written = 0
        with wds.ShardWriter(
            args.out_pattern, maxcount=args.maxcount, maxsize=args.maxsize
        ) as sink:
            for m in members:
                data = tar.extractfile(m).read()
                key = sample_key(m.name)
                if args.prefix and key.startswith(args.prefix):
                    key = key[len(args.prefix):]
                sink.write({"__key__": key, "jpeg": data})
                written += 1
                if written % 50000 == 0:
                    print(f"  {written}/{len(members)}")

    print(f"Done: wrote {written} samples to shards matching {args.out_pattern}")


if __name__ == "__main__":
    main()
