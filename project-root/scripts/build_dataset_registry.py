"""
build_dataset_registry.py
============================

SAD §6.2 Data Storage Schema: builds `dataset_registry.csv` and
`train.csv`/`val.csv`/`test.csv` from data that already exists in this
repo (`data/splits.json` for per-class train/val/test membership,
`data/logo_boxes.json` for which images have a planted shortcut).

Nothing in the project previously materialised these files — splits.json
was read ad hoc, and there was no single registry describing the dataset
as a whole. This script is the one place that schema gets built, so
downstream code (ImageDataLoader, dashboards, experiment logs) can all
point at the same source of truth.

Deviates from the SAD's literal folder layout: outputs go to
`data/processed/{train,val,test}.csv` and `data/metadata/dataset_registry.csv`
as documented in §6.2's folder tree — that part IS followed. label_id is
a synthetic 0..7 index (sorted class name), not an ImageNet synset id,
since `data/splits.json`'s class set doesn't match
`src/models/clip_model.py`'s TRUCK_CLASSES synset mapping 1:1 (e.g.
"beer_truck" has no synset entry there) — see that file's docstring.

Usage
-----
    python scripts/build_dataset_registry.py \
        --splits-path data/splits.json \
        --logo-boxes-path data/logo_boxes.json \
        --dataset-id imagenet_trucks \
        --image-size 224x224
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SPLIT_NAMES = ["train", "val", "test"]


def build_registry_row(dataset_id: str, name: str, splits: dict, image_size: str, source_url: str) -> dict:
    num_classes = len(splits)
    num_samples = sum(
        len(files) for split_map in splits.values() for files in split_map.values()
    )
    return {
        "dataset_id": dataset_id,
        "name": name,
        "task_type": "classification",
        "num_classes": num_classes,
        "num_samples": num_samples,
        "image_size": image_size,
        "split_path": "data/processed/",
        "source_url": source_url,
    }


def build_split_csvs(dataset_id: str, splits: dict, logo_boxes: dict) -> dict:
    class_names = sorted(splits.keys())
    label_id_by_class = {name: idx for idx, name in enumerate(class_names)}

    rows_by_split = {split: [] for split in SPLIT_NAMES}

    for class_name in class_names:
        split_map = splits[class_name]
        for split in SPLIT_NAMES:
            for filename in split_map.get(split, []):
                rows_by_split[split].append(
                    {
                        "sample_id": f"{class_name}/{filename}",
                        "image_path": f"data/clean/{class_name}/{filename}",
                        "dataset_id": dataset_id,
                        "label": class_name,
                        "label_id": label_id_by_class[class_name],
                        "split": split,
                        "has_shortcut": filename in logo_boxes,
                        "notes": "",
                    }
                )

    return {split: pd.DataFrame(rows) for split, rows in rows_by_split.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-path", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--logo-boxes-path", type=Path, default=Path("data/logo_boxes.json"))
    parser.add_argument("--dataset-id", default="imagenet_trucks")
    parser.add_argument("--name", default="ImageNet Truck Subset")
    parser.add_argument("--image-size", default="224x224")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/metadata"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    with open(args.splits_path, "r") as f:
        splits = json.load(f)

    logo_boxes = {}
    if args.logo_boxes_path.exists():
        with open(args.logo_boxes_path, "r") as f:
            logo_boxes = json.load(f)

    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    registry_row = build_registry_row(args.dataset_id, args.name, splits, args.image_size, args.source_url)
    registry_path = args.metadata_dir / "dataset_registry.csv"
    pd.DataFrame([registry_row]).to_csv(registry_path, index=False)
    print(f"Wrote {registry_path}")

    split_dfs = build_split_csvs(args.dataset_id, splits, logo_boxes)
    for split, df in split_dfs.items():
        out_path = args.processed_dir / f"{split}.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(df)} rows, {int(df['has_shortcut'].sum()) if len(df) else 0} flagged has_shortcut)")


if __name__ == "__main__":
    main()
