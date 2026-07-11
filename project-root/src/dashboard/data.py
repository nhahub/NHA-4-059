"""
data.py
========

Reads pre-generated result artifacts (PNGs + the three-way comparison CSV)
from project-root/results/, which the notebook's push-cell keeps in sync
with whatever it last generated on Drive. This dashboard is a "browse
precomputed results" tool, not a live-inference one -- there's no local
CLIP/model dependency here at all, only files on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# (filename, caption) pairs per model, in display order. `available_images`
# filters these down to whatever actually exists in results/ at request
# time, so a tab renders whatever the notebook has generated so far instead
# of erroring on files it hasn't produced yet.
RN50_IMAGES: List[Tuple[str, str]] = [
    ("01_dataset_distribution.png", "Dataset distribution (train/test)"),
    ("02_sample_images.png", "Sample image per class"),
    ("03_classification_results.png", "Confusion matrix + per-class accuracy"),
    ("04_logo_insertion_examples.png", "Logo insertion examples"),
    ("05_logo_comparison.png", "Per-class accuracy: original vs logo"),
    ("06_final_results_table.png", "Detection + mitigation summary table"),
    ("07_final_pipeline_chart.png", "Detection + mitigation pipeline chart"),
    ("08_gradcam_all_classes.png", "Grad-CAM, all classes (clean images)"),
    ("09_gradcam_logo_comparison.png", "Grad-CAM before/after logo"),
    ("10_bilrp_garbage.png", "BiLRP: garbage truck ↔ garbage truck"),
    ("11_bilrp_pickup.png", "BiLRP: pickup ↔ pickup"),
    ("12_bilrp_logo_shift.png", "BiLRP: original ↔ same image + logo"),
    ("10b_bilrp_garbage_naive_comparison.png", "BiLRP: naive (buggy) method, for comparison"),
    ("13_gradcam_vs_lrp.png", "Grad-CAM vs LRP"),
    ("14_gradcam_early_layer.png", "Early-layer (bn3) Grad-CAM"),
]

VIT_IMAGES: List[Tuple[str, str]] = [
    ("24_vit_logo_comparison.png", "Per-class accuracy: original vs logo"),
    ("25_vit_attention_rollout.png", "Attention Rollout before/after logo"),
    ("26_vit_bilrp_logo_shift.png", "BiLRP: original ↔ same image + logo"),
    ("27_vit_final_pipeline_chart.png", "Detection + mitigation pipeline chart"),
]

SUPERVISED_RESNET_IMAGES: List[Tuple[str, str]] = [
    ("28_supervised_resnet_logo_comparison.png", "Per-class accuracy: original vs logo"),
    ("29_supervised_resnet_gradcam.png", "Grad-CAM before/after logo"),
    ("30_supervised_resnet_bilrp.png", "BiLRP: original ↔ same image + logo"),
    ("31_supervised_resnet_gradcam_vs_lrp.png", "Grad-CAM vs LRP"),
    ("32_supervised_resnet_final_pipeline_chart.png", "Detection + mitigation pipeline chart"),
]


def available_images(image_list: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Filter (filename, caption) pairs down to ones that actually exist in
    results/ right now."""
    return [(fname, caption) for fname, caption in image_list if (RESULTS_DIR / fname).exists()]


def load_comparison_table() -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / "three_way_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)
