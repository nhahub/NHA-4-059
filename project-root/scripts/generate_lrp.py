"""
generate_lrp.py
================

FR-7: generates Grad-CAM and LRP heatmaps for the same image set and
computes their spatial correlation. Mirrors generate_heatmaps.py's CLI
and conventions, but computes both explainers per image so they can be
compared directly (each explains the same predicted class, via the same
classifier logit target — see src/xai/gradcam.py and src/xai/lrp.py).

Outputs
-------
    outputs/heatmaps_lrp/{method}/{class}/{stem}.png     LRP overlay PNGs
    outputs/metrics/gradcam_lrp_correlation.csv
        columns: filename, class, method, spatial_correlation,
                 gradcam_energy_ratio, lrp_energy_ratio

Usage
-----
    python scripts/generate_lrp.py \
        --classifier-path /path/to/clf_full.pkl \
        --examples-path   /path/to/truck_samples_test_FULL.pkl \
        --methods clean blur replace crop
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.clip_model import CLIPModel, decode_image, TRUCK_CLASSES
from src.data.image_modifier import ImageModifier
from src.xai.gradcam import GradCAM, attention_alignment_score
from src.xai.lrp import LRPExplainer, spatial_correlation
from src.utils.experiment_config import resolve_config, set_seed
from src.utils.experiment_log import ExperimentLogger

METHODS = ["clean", "blur", "replace", "crop"]


def run_variant(
    clip_model: CLIPModel,
    gradcam: GradCAM,
    lrp: LRPExplainer,
    modifier: ImageModifier,
    examples: list,
    method: str,
    output_dir: Path,
) -> pd.DataFrame:
    label_to_name = {v: k for k, v in TRUCK_CLASSES.items()}
    rows = []

    for idx, example in enumerate(tqdm(examples, desc=f"Grad-CAM+LRP [{method}]")):
        img = decode_image(example["image"])
        if method != "clean":
            img = modifier.apply(img, method)
        img_224 = img.convert("RGB").resize((224, 224))

        cam, class_idx = gradcam(img_224)
        lrp_map, _ = lrp(img_224, class_index_in_clf=class_idx)

        boxes = modifier.logo_regions(img_224)
        gradcam_score = attention_alignment_score(cam, boxes)
        lrp_score = attention_alignment_score(lrp_map, boxes)
        corr = spatial_correlation(cam, lrp_map)

        filename = example.get("filename", f"img_{idx:05d}.jpg")
        true_class = label_to_name.get(example["label"], str(example["label"]))
        stem = Path(filename).stem

        class_dir = output_dir / "heatmaps_lrp" / method / true_class
        class_dir.mkdir(parents=True, exist_ok=True)
        overlay_img = gradcam.overlay_heatmap(img_224, lrp_map)
        Image.fromarray(overlay_img).save(class_dir / f"{stem}.png")

        rows.append(
            {
                "filename": filename,
                "class": true_class,
                "method": method,
                "spatial_correlation": round(corr, 4),
                "gradcam_energy_ratio": round(gradcam_score, 4),
                "lrp_energy_ratio": round(lrp_score, 4),
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
    lrp = LRPExplainer(clip_model)
    modifier = ImageModifier()

    with open(config.examples_path, "rb") as f:
        examples = pickle.load(f)

    all_rows = []
    for method in config.methods:
        df = run_variant(clip_model, gradcam, lrp, modifier, examples, method, output_dir)
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    out_path = metrics_dir / "gradcam_lrp_correlation.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(combined)} rows)")
    print(f"Mean spatial correlation: {combined['spatial_correlation'].mean():.4f}")

    logger = ExperimentLogger(output_dir / "experiment_log.csv")
    exp_id = logger.log(
        model_name=config.model_name,
        dataset_id="imagenet_trucks",
        xai_method="gradcam_vs_lrp",
        accuracy=float("nan"),
        focus_score=float(combined["spatial_correlation"].mean()) if len(combined) else float("nan"),
        shortcut_detected=bool((combined["lrp_energy_ratio"] > 0.30).any()) if len(combined) else False,
        heatmap_path=str(output_dir / "heatmaps_lrp"),
        notes=config.notes,
    )
    print(f"Logged experiment {exp_id} to {output_dir / 'experiment_log.csv'}")


if __name__ == "__main__":
    main()
