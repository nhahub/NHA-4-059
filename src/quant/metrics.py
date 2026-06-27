"""
Standard classification metrics on CLIP's zero-shot predictions.

Owner: Person D (Quant Analysis Lead)
Day: 2

Run: python src/quant/metrics.py --dataset original
     python src/quant/metrics.py --dataset logo_variant

Consumes: outputs/predictions/<dataset>_predictions.parquet (from Person A)
Output:   outputs/figures/<dataset>_confusion_matrix.png
          printed accuracy / precision / recall / F1 (overall + per class)
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def compute_metrics(dataset_split: str, config: dict):
    pred_path = Path(config["paths"]["predictions_out"]) / f"{dataset_split}_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} not found — run src/clip_inference/run_zero_shot.py "
            f"--dataset {dataset_split} first (Person A's output)."
        )
    df = pd.read_parquet(pred_path)

    print(f"\n=== Metrics for '{dataset_split}' (n={len(df)}) ===")
    print(f"Overall accuracy: {df['correct'].mean():.4f}\n")

    print(classification_report(df["true_class"], df["predicted_class"]))

    class_names = list(config["classes"].keys())
    cm = confusion_matrix(df["true_class"], df["predicted_class"], labels=class_names)

    fig_dir = Path(config["paths"]["figures_out"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {dataset_split}")
    plt.tight_layout()
    out_path = fig_dir / f"{dataset_split}_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved to {out_path}")

    return df, cm


def compute_accuracy_delta(original_df: pd.DataFrame, logo_df: pd.DataFrame):
    """
    The core hypothesis test: does accuracy drop when a logo is inserted?
    This is the project's central quantitative claim — keep this number
    front and center in the report and dashboard.
    """
    orig_acc = original_df.groupby("true_class")["correct"].mean()
    logo_acc = logo_df.groupby("true_class")["correct"].mean()

    delta = (logo_acc - orig_acc).sort_values()
    print("\n=== Accuracy delta: logo-inserted minus original (per class) ===")
    print(delta)
    print(f"\nOverall: {original_df['correct'].mean():.4f} -> {logo_df['correct'].mean():.4f} "
          f"(delta: {logo_df['correct'].mean() - original_df['correct'].mean():+.4f})")
    return delta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["original", "logo_variant", "both"], default="original")
    args = parser.parse_args()

    config = load_config()

    if args.dataset == "both":
        orig_df, _ = compute_metrics("original", config)
        logo_df, _ = compute_metrics("logo_variant", config)
        compute_accuracy_delta(orig_df, logo_df)
    else:
        compute_metrics(args.dataset, config)
