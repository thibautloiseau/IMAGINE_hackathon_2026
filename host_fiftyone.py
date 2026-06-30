#!/usr/bin/env python3
"""
Launch FiftyOne with a dataset directory.

Usage:
    python3 host_fiftyone.py /path/to/dataset
"""
import os
import sys
from pathlib import Path

os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,::1"

import fiftyone as fo

PORT = 8080
ADDRESS = "0.0.0.0"

if len(sys.argv) < 2:
    print(__doc__.strip())
    sys.exit(1)

data_dir = Path(sys.argv[1])

if not data_dir.exists():
    print(f"Error: {data_dir} does not exist.")
    sys.exit(1)

if not any(data_dir.rglob("*.[jJ][pP][gG]")) and not any(data_dir.rglob("*.[jJ][pP][eE][gG]")):
    print(f"Error: {data_dir} contains no images.")
    sys.exit(1)

# Auto-detect: subdirectories → classification tree, else flat folder
subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
dataset_type = (
    fo.types.ImageClassificationDirectoryTree if subdirs
    else fo.types.ImageDirectory
)

DATASET_NAME = "imagenet_val"

# Use a fixed name so it reuses the already-imported dataset
if DATASET_NAME in fo.list_datasets():
    dataset = fo.load_dataset(DATASET_NAME)
    print(f"Reusing existing dataset: {dataset.name} with {len(dataset)} samples")
else:
    dataset = fo.Dataset(name=DATASET_NAME)
    dataset.add_dir(str(data_dir), dataset_type=dataset_type)
    print(f"Dataset: {dataset.name}  |  Samples: {len(dataset)}")

    # Map ImageNet synset IDs to human-readable class names
    if subdirs:
        _here = Path(__file__).resolve().parent
        _synsets_file = _here / ".venv" / "lib" / "python3.12" / "site-packages" / "eta" / "resources" / "imagenet-labels.txt"
        _synset_list_file = Path.home() / ".cache" / "uv" / "archive-v0" / "L0m7V0P9-00DFl61" / "timm" / "data" / "_info" / "imagenet_synsets.txt"
        if _synset_list_file.exists() and _synsets_file.exists():
            with open(_synset_list_file) as f:
                _synsets = [l.strip() for l in f]
            with open(_synsets_file) as f:
                _labels = [l.strip().split(":", 1)[1] for l in f]
            _synset_to_name = dict(zip(_synsets, _labels[1:]))
            # Get current labels (synset IDs) and map them to class names
            labels = dataset.values("ground_truth.label")
            class_names = [_synset_to_name.get(l, l) for l in labels]
            dataset.set_values("ground_truth.label", class_names)
            dataset.save()
            print("Labels mapped to human-readable class names")

session = fo.launch_app(dataset, port=PORT, address=ADDRESS)
session.wait()
