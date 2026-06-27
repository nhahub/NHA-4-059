"""
Download and organize the 8-class ImageNet truck subset.
Owner: Person C (Data Engineer)
Day: 1

Run: python src/data_prep/download_dataset.py
Output: data/raw/<class_name>/<image>.jpg for each of the 8 classes

Source options (pick one based on access — Hugging Face is usually the
easiest, no Kaggle API key needed):
  1. Hugging Face: "imagenet-1k" dataset, filtered by synset ID
  2. torchvision.datasets.ImageNet (requires manual ImageNet download + license)
  3. Kaggle ImageNet subset (requires kaggle API credentials)

This script scaffolds option 1. Swap in option 2/3 if HF access is blocked.
"""
import os
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def download_via_huggingface(config):
    """
    Pulls images for each target synset from the imagenet-1k HF dataset.
    Requires: pip install datasets, and `huggingface-cli login` if the
    dataset is gated (imagenet-1k usually requires accepting terms once).
    """
    from datasets import load_dataset

    raw_dir = Path(config["paths"]["raw_data"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    target_synsets = {v["synset"]: k for k, v in config["classes"].items()}
    print(f"Looking for {len(target_synsets)} target synsets: {list(target_synsets.keys())}")

    # NOTE: imagenet-1k on HF is indexed by integer label, not synset string
    # directly in the basic config. You will likely need the label->synset
    # mapping file (LOC_synset_mapping.txt equivalent) to filter correctly.
    # This is flagged as a Day 1 task: confirm the exact filtering mechanism
    # for whichever source you actually use, and update this function.

    raise NotImplementedError(
        "TODO (Person C, Day 1): confirm dataset source access (HF gated "
        "dataset approval, Kaggle credentials, or local ImageNet download) "
        "and implement the actual filtering + save-to-disk logic here. "
        "Target: ~50+ images per class minimum for a usable test set."
    )


def verify_dataset_structure(config):
    """Sanity check: confirm every class folder exists and has images."""
    raw_dir = Path(config["paths"]["raw_data"])
    report = {}
    for class_name in config["classes"]:
        class_dir = raw_dir / class_name
        n_images = len(list(class_dir.glob("*.jpg"))) if class_dir.exists() else 0
        report[class_name] = n_images
        status = "OK" if n_images >= 30 else "LOW / MISSING"
        print(f"  {class_name:20s} {n_images:4d} images  [{status}]")
    return report


if __name__ == "__main__":
    config = load_config()
    print("Target classes:", list(config["classes"].keys()))
    # download_via_huggingface(config)   # uncomment once implemented
    verify_dataset_structure(config)
