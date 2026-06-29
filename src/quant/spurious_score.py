"""
Spurious-feature reliance score: what fraction of Grad-CAM relevance falls
inside the known logo bounding box vs. on the rest of the image (presumably
the vehicle body)?

This turns "the heatmap looks like it's focusing on the logo" into a single
number per image that can be aggregated, charted, and compared across
classes — this IS objective 5 in the project doc ("quantify the extent of
spurious-feature dependency").

Owner: Person D (Quant Analysis Lead)
Day: 3

Consumes: outputs/heatmaps/<dataset>/<class>/<image>.npy   (from Person B)
          data/annotations/logo_boxes.json                  (from Person C)
Output:   outputs/predictions/<dataset>_spurious_scores.parquet
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def heatmap_energy_ratio(heatmap: np.ndarray, bbox: dict) -> float:
    """
    Returns the fraction of total positive heatmap energy that falls inside
    the logo bounding box. A score near 0 means the model ignored the logo
    region; a score much higher than the box's area fraction of the image
    means the model is disproportionately relying on it.
    """
    heatmap = np.clip(heatmap, a_min=0, a_max=None)  # only positive relevance
    total_energy = heatmap.sum()
    if total_energy == 0:
        return 0.0

    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    box_energy = heatmap[y:y + h, x:x + w].sum()
    return float(box_energy / total_energy)


def compute_scores(dataset_split: str, config: dict):
    """
    Only meaningful for the logo_variant dataset (or for garbage_truck/
    beer_truck images in the original set if you've separately annotated
    real logo locations there — see data/annotations/logo_boxes.json).
    """
    heatmap_dir = Path(config["paths"]["heatmaps_out"]) / dataset_split
    ann_path = Path(config["paths"]["annotations"]) / "logo_boxes.json"

    if not ann_path.exists():
        raise FileNotFoundError(
            f"{ann_path} not found — run src/data_prep/insert_logo.py first "
            "(Person C's output), or manually annotate real logo boxes for "
            "the original dataset if scoring that instead."
        )
    with open(ann_path) as f:
        boxes = json.load(f)

    records = []
    for class_dir in sorted(heatmap_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for heatmap_path in sorted(class_dir.glob("*.npy")):
            # match heatmap back to its corresponding image path used as the
            # key in logo_boxes.json — adjust this lookup if your naming
            # convention differs between data_dir and heatmap_dir
            image_key_candidates = [
                k for k in boxes if Path(k).stem == heatmap_path.stem
                and class_name in k
            ]
            if not image_key_candidates:
                continue
            bbox = boxes[image_key_candidates[0]]

            heatmap = np.load(heatmap_path)
            score = heatmap_energy_ratio(heatmap, bbox)
            records.append({
                "class_name": class_name,
                "image_stem": heatmap_path.stem,
                "spurious_score": score,
            })

    df = pd.DataFrame(records)
    out_path = Path(config["paths"]["predictions_out"]) / f"{dataset_split}_spurious_scores.parquet"
    df.to_parquet(out_path)

    print(f"\n=== Spurious-feature reliance scores ({dataset_split}) ===")
    print(df.groupby("class_name")["spurious_score"].agg(["mean", "std", "count"]))
    print(f"\nSaved to {out_path}")
    return df


if __name__ == "__main__":
    config = load_config()
    compute_scores("logo_variant", config)
