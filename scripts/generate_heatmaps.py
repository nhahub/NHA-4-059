"""
generate_heatmaps.py
=====================

Missing link flagged in review: nothing in the project actually ran
GradCAM end-to-end and saved PNG heatmaps — `src/xai/gradcam.py` existed,
but no script called it, overlaid the result on the source image, or wrote
anything to `outputs/heatmaps/`, even though the dashboard
(`src/dashboard/callbacks.py`) already reads heatmaps from exactly that
path and reads an `outputs/metrics/energy_ratio_all.csv` for the FR-5
Attention Alignment Score.

This script closes that gap: for each modification method (clean / blur /
replace / crop), it runs GradCAM on every example, saves a heatmap overlay
PNG, and computes the Attention Alignment Score — the fraction of the
CAM's energy that falls inside the known logo regions (bottom-left +
top-right, per `ImageModifier.logo_regions`) — flagging anything over the
FR-5 30% spurious-reliance threshold.

Outputs
-------
    outputs/heatmaps/{method}/{class}/{stem}.png
    outputs/metrics/energy_ratio_all.csv
        columns: filename, class, method, energy_ratio
    outputs/metrics/spurious_flags.csv
        rows from the above where energy_ratio > 0.30

Usage
-----
    python scripts/generate_heatmaps.py \
        --classifier-path /path/to/clf_full.pkl \
        --examples-path   /path/to/truck_samples_test_FULL.pkl \
        --methods clean blur replace crop
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.clip_model import CLIPModel, decode_image, TRUCK_CLASSES
from src.data.image_modifier import ImageModifier
from src.xai.gradcam import GradCAM, attention_alignment_score
from src.utils.experiment_config import ExperimentConfig, resolve_config, set_seed
from src.utils.experiment_log import ExperimentLogger

METHODS = ["clean", "blur", "replace", "crop"]
ALIGNMENT_FLAG_THRESHOLD = 0.30  # FR-5: flag if >30% of attention falls on spurious regions


def run_variant(
    clip_model: CLIPModel,
    gradcam: GradCAM,
    modifier: ImageModifier,
    examples: list,
    method: str,
    output_dir: Path,
) -> pd.DataFrame:
    label_to_name = {v: k for k, v in TRUCK_CLASSES.items()}
    rows = []

    for idx, example in enumerate(tqdm(examples, desc=f"GradCAM [{method}]")):
        img = decode_image(example["image"])
        if method != "clean":
            img = modifier.apply(img, method)

        # Resize once to CLIP's 224x224 input grid so the logo boxes (computed
        # on this same resized image) land on the same pixel grid as the CAM.
        img_224 = img.convert("RGB").resize((224, 224))

        cam, class_idx = gradcam(img_224)
        boxes = modifier.logo_regions(img_224)
        score = attention_alignment_score(cam, boxes)

        filename = example.get("filename", f"img_{idx:05d}.jpg")
        true_class = label_to_name.get(example["label"], str(example["label"]))
        stem = Path(filename).stem

        class_dir = output_dir / "heatmaps" / method / true_class
        class_dir.mkdir(parents=True, exist_ok=True)

        overlay = gradcam.overlay_heatmap(img_224, cam)
        Image.fromarray(overlay).save(class_dir / f"{stem}.png")

        rows.append(
            {
                "filename": filename,
                "class": true_class,
                "method": method,
                "energy_ratio": round(score, 4),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-path", type=Path, required=False, help="Path to clf_full.pkl")
    parser.add_argument("--examples-path", type=Path, required=False, help="Path to truck_samples_test_FULL.pkl")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42, help="Random seed, for FR-8 reproducibility")
    parser.add_argument("--config", type=Path, default=None, help="Load an ExperimentConfig JSON instead of the flags above")
    parser.add_argument("--save-config", type=Path, default=None, help="Save the resolved ExperimentConfig JSON here for later reuse")
    args = parser.parse_args()

    if args.config is None and (args.classifier_path is None or args.examples_path is None):
        parser.error("--classifier-path and --examples-path are required unless --config is given")

    config = resolve_config(args, args.config)
    if args.save_config:
        config.save(str(args.save_config))
        print(f"Saved experiment config to {args.save_config}")

    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    clip_model = CLIPModel(classifier_path=config.classifier_path)
    gradcam = GradCAM(clip_model)
    modifier = ImageModifier()

    with open(config.examples_path, "rb") as f:
        examples = pickle.load(f)

    all_rows = []
    for method in config.methods:
        df = run_variant(clip_model, gradcam, modifier, examples, method, output_dir)
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    energy_path = metrics_dir / "energy_ratio_all.csv"
    combined.to_csv(energy_path, index=False)
    print(f"Wrote {energy_path} ({len(combined)} rows)")

    flagged = combined[combined["energy_ratio"] > ALIGNMENT_FLAG_THRESHOLD]
    flagged_path = metrics_dir / "spurious_flags.csv"
    flagged.to_csv(flagged_path, index=False)
    print(f"Wrote {flagged_path} ({len(flagged)} rows flagged, >{ALIGNMENT_FLAG_THRESHOLD:.0%} attention on logo regions)")

    logger = ExperimentLogger(output_dir / "experiment_log.csv")
    exp_id = logger.log(
        model_name=config.model_name,
        dataset_id="imagenet_trucks",
        xai_method="gradcam",
        accuracy=float("nan"),
        focus_score=float(combined["energy_ratio"].mean()) if len(combined) else float("nan"),
        shortcut_detected=bool(len(flagged) > 0),
        heatmap_path=str(output_dir / "heatmaps"),
        notes=config.notes,
    )
    print(f"Logged experiment {exp_id} to {output_dir / 'experiment_log.csv'}")


if __name__ == "__main__":
    main()
