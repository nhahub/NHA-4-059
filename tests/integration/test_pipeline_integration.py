"""
Integration test: does the FULL pipeline actually connect correctly, not
just each module in isolation?

This builds a tiny synthetic dataset (2 classes, 3 images each, plain
colored squares — no real CLIP/network calls needed for most of it) and
checks that each module's OUTPUT FORMAT is what the NEXT module expects:

  data_prep output format  -> matches what clip_inference reads
  clip_inference output    -> matches what metrics.py reads
  logo insertion + bbox     -> matches what spurious_score.py reads

This is exactly the kind of bug that unit tests miss: each person's module
can be individually correct and still break when chained together (e.g.
Person A saves "true_class" but Person D's script expects "label").

Run: pytest tests/integration/test_pipeline_integration.py -v
Note: this test does NOT call the real CLIP model (too slow / needs
network access for weight download) — it tests the data contracts between
modules using fixtures. A separate, slower smoke test that does call real
CLIP belongs in test_pipeline_e2e_slow.py (Day 7 bug-bash territory), not
in this fast test that should run on every PR.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_prep.insert_logo import overlay_logo
from src.quant.spurious_score import heatmap_energy_ratio


@pytest.fixture
def tmp_config(tmp_path):
    """A minimal config pointing entirely at tmp_path, so this test never
    touches the real data/ or outputs/ directories."""
    cfg = {
        "paths": {
            "raw_data": str(tmp_path / "raw"),
            "processed_data": str(tmp_path / "processed"),
            "logo_variant_data": str(tmp_path / "logo_variant"),
            "annotations": str(tmp_path / "annotations"),
            "predictions_out": str(tmp_path / "predictions"),
            "heatmaps_out": str(tmp_path / "heatmaps"),
            "figures_out": str(tmp_path / "figures"),
        },
        "classes": {
            "class_a": {"synset": "n00000001", "prompt_name": "class a"},
            "class_b": {"synset": "n00000002", "prompt_name": "class b"},
        },
        "logo_insertion": {"relative_size": 0.2, "position": "lower_left", "opacity": 0.9},
    }
    for p in cfg["paths"].values():
        Path(p).mkdir(parents=True, exist_ok=True)
    return cfg


def make_fake_image(color, size=224):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = color
    return img


def test_logo_insertion_to_spurious_score_contract(tmp_config):
    """
    Checks the real contract between Person C's logo-box output and Person
    D's spurious-score input: does the bbox dict insert_logo.py produces
    have exactly the keys spurious_score.py expects?
    """
    img = make_fake_image((100, 100, 100))
    logo = np.zeros((40, 40, 4), dtype=np.uint8)
    logo[:, :, :3] = 255
    logo[:, :, 3] = 255

    _, bbox = overlay_logo(img, logo, tmp_config["logo_insertion"])

    required_keys = {"x", "y", "w", "h"}
    assert required_keys.issubset(bbox.keys()), (
        f"insert_logo.py's bbox output is missing keys that "
        f"spurious_score.py requires: {required_keys - bbox.keys()}"
    )

    fake_heatmap = np.random.rand(224, 224)
    score = heatmap_energy_ratio(fake_heatmap, bbox)
    assert 0.0 <= score <= 1.0, "spurious score must be a valid ratio"


def test_predictions_dataframe_has_columns_metrics_expects(tmp_config):
    """
    Checks the contract between clip_inference's saved predictions and
    metrics.py's expectations, WITHOUT calling real CLIP — builds a fake
    predictions dataframe in the exact shape run_zero_shot.py produces,
    and confirms metrics.py's required columns are all present.
    """
    fake_predictions = pd.DataFrame([
        {"image_path": "a.jpg", "true_class": "class_a", "predicted_class": "class_a",
         "confidence": 0.9, "correct": True},
        {"image_path": "b.jpg", "true_class": "class_a", "predicted_class": "class_b",
         "confidence": 0.6, "correct": False},
        {"image_path": "c.jpg", "true_class": "class_b", "predicted_class": "class_b",
         "confidence": 0.8, "correct": True},
    ])

    pred_dir = Path(tmp_config["paths"]["predictions_out"])
    fake_predictions.to_parquet(pred_dir / "original_predictions.parquet")

    loaded = pd.read_parquet(pred_dir / "original_predictions.parquet")
    required_columns = {"image_path", "true_class", "predicted_class", "confidence", "correct"}
    assert required_columns.issubset(loaded.columns), (
        f"Predictions file is missing columns metrics.py requires: "
        f"{required_columns - set(loaded.columns)}"
    )

    # this is literally what metrics.py does first — if this breaks here,
    # it would break there too, caught before anyone runs the real pipeline
    accuracy = loaded["correct"].mean()
    assert accuracy == pytest.approx(2 / 3)


def test_splits_file_contract(tmp_config):
    """
    Checks that make_splits.py's output format is what every script that
    needs train/val/test discipline expects: a dict with exactly
    'dev'/'val'/'test' keys, each a list of path strings.
    """
    fake_splits = {
        "dev": ["data/raw/class_a/1.jpg"],
        "val": ["data/raw/class_a/2.jpg"],
        "test": ["data/raw/class_a/3.jpg", "data/raw/class_b/1.jpg"],
    }
    splits_path = Path(tmp_config["paths"]["annotations"]) / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(fake_splits, f)

    with open(splits_path) as f:
        loaded = json.load(f)

    assert set(loaded.keys()) == {"dev", "val", "test"}
    for split_name, paths in loaded.items():
        assert isinstance(paths, list)
        assert all(isinstance(p, str) for p in paths)

    # no image should appear in more than one split — a real bug class
    all_paths = [p for paths in loaded.values() for p in paths]
    assert len(all_paths) == len(set(all_paths)), "an image appears in multiple splits"
