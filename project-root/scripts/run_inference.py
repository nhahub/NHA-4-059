"""
run_inference.py
=================

Missing link in the pipeline: accuracy_delta.py and compute_metrics.py both
expect `outputs/inference/inference_{clean,blur,replace,crop}.csv`, but
nothing in the original notebook or project-root ever wrote them — the
notebook computed accuracy inline with numpy arrays and never serialized
per-image predictions to CSV.

This script runs the classifier over a test set under each modification
(clean / blur / replace / crop) and writes one CSV per variant with the
columns accuracy_delta.py already knows how to read:
    filename, true_class, predicted_class

Usage
-----
    python scripts/run_inference.py \
        --classifier-path /path/to/clf_full.pkl \
        --examples-path   /path/to/truck_samples_test_FULL.pkl \
        --output-dir outputs/inference
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.clip_model import CLIPModel, decode_image, TRUCK_CLASSES
from src.data.image_modifier import ImageModifier

METHODS = ["clean", "blur", "replace", "crop"]


def run_variant(clip_model: CLIPModel, modifier: ImageModifier, examples: list, method: str) -> pd.DataFrame:
    label_to_name = {v: k for k, v in TRUCK_CLASSES.items()}
    rows = []

    for idx, example in enumerate(tqdm(examples, desc=f"Inference [{method}]")):
        img = decode_image(example["image"])
        if method != "clean":
            img = modifier.apply(img, method)

        feat = clip_model.encode_image(img)
        pred_idx = int(clip_model.predict(feat)[0])

        rows.append(
            {
                "filename": example.get("filename", f"img_{idx:05d}.jpg"),
                "true_class": label_to_name.get(example["label"], str(example["label"])),
                "predicted_class": label_to_name.get(pred_idx, str(pred_idx)),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-path", type=Path, required=True, help="Path to clf_full.pkl")
    parser.add_argument("--examples-path", type=Path, required=True, help="Path to truck_samples_test_FULL.pkl")
    parser.add_argument("--logo-boxes-path", type=Path, default=None, help="Optional data/logo_boxes.json")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    clip_model = CLIPModel(classifier_path=args.classifier_path)
    modifier = ImageModifier(logo_boxes_path=args.logo_boxes_path)

    with open(args.examples_path, "rb") as f:
        examples = pickle.load(f)

    for method in METHODS:
        df = run_variant(clip_model, modifier, examples, method)
        out_path = args.output_dir / f"inference_{method}.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
