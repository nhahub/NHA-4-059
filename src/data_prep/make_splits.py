"""
Split the dataset into train/val/test BEFORE anyone touches accuracy numbers.

This isn't optional plumbing — it's what makes the project's quantitative
claims defensible. Without a held-out split:
  - "tuning" prompt templates or logo-insertion params against the same
    images you report accuracy on silently inflates your numbers
  - there's no way to sanity-check that the accuracy-delta finding
    generalizes rather than being an artifact of the specific images chosen

We don't need a "train" set in the traditional sense (CLIP isn't being
trained), but we still split three ways so the team has discipline around
which images are used for what:

  - dev/   (~15%) — use freely while building scripts, debugging Grad-CAM,
                    eyeballing heatmaps. Numbers from this set are NOT
                    reported in the final results.
  - val/   (~15%) — use ONLY to make decisions: e.g. confirming the
                    logo-insertion size/position/opacity actually produces
                    a visible accuracy effect before locking those params,
                    or choosing between ViT/RN50 if both are viable.
                    Lock all such decisions by Day 4. After that, val/ is
                    not touched again.
  - test/  (~70%) — touched exactly once, for the numbers that go in the
                    report and dashboard. If a bug is found after first use,
                    document it — don't quietly re-run until the number
                    looks right.

Owner: Person C (Data Engineer)
Day: 1-2, right after raw download, before anyone runs inference

Run: python src/data_prep/make_splits.py
Output: data/annotations/splits.json  { "dev": [...], "val": [...], "test": [...] }
"""
import json
import random
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
SEED = 42  # fixed seed so the split is reproducible across the team


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def make_splits(config: dict, dev_frac=0.15, val_frac=0.15):
    raw_dir = Path(config["paths"]["raw_data"])
    rng = random.Random(SEED)

    splits = {"dev": [], "val": [], "test": []}

    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images = sorted(class_dir.glob("*.jpg"))
        rng.shuffle(images)

        n = len(images)
        if n < 5:
            print(f"  WARNING: {class_dir.name} has only {n} images — too "
                  "few to split meaningfully. Get more data before trusting "
                  "any split-based numbers for this class.")
        n_dev = max(1, int(n * dev_frac)) if n > 0 else 0
        n_val = max(1, int(n * val_frac)) if n > 1 else 0
        # guard against dev+val exceeding n on small classes
        n_dev = min(n_dev, n)
        n_val = min(n_val, max(0, n - n_dev))

        splits["dev"].extend(str(p) for p in images[:n_dev])
        splits["val"].extend(str(p) for p in images[n_dev:n_dev + n_val])
        splits["test"].extend(str(p) for p in images[n_dev + n_val:])

        print(f"  {class_dir.name:20s} total={n:4d}  dev={n_dev:3d}  "
              f"val={n_val:3d}  test={n - n_dev - n_val:3d}")

    out_path = Path(config["paths"]["annotations"]) / "splits.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\nSplit sizes — dev: {len(splits['dev'])}, "
          f"val: {len(splits['val'])}, test: {len(splits['test'])}")
    print(f"Saved to {out_path}")
    return splits


def load_splits(config: dict) -> dict:
    """Used by every downstream script that needs to know which images
    belong to which split — import this rather than re-implementing."""
    path = Path(config["paths"]["annotations"]) / "splits.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run make_splits.py first. "
            "All inference/XAI/quant scripts depend on this."
        )
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    config = load_config()
    make_splits(config)
