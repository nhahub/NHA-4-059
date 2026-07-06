"""
accuracy_delta.py
==================

Owner: Abdelrahman (scripts/ — Metrics & Analysis)

Purpose
-------
Quantifies "Clever Hans" / shortcut-learning behaviour by comparing CLIP's
per-class accuracy and F1 score on CLEAN images against three modified
versions of the same images (logo blurred, logo replaced, logo cropped).

A large drop in accuracy after a logo-only modification implies the model
was relying on the logo (a spurious feature) rather than the truck body
itself to make its prediction.

Inputs (CSV files, produced upstream by run_inference.py / Catherine)
-----------------------------------------------------------------------
    inference_clean.csv
    inference_blur.csv
    inference_replace.csv
    inference_crop.csv

Each file is expected to contain, at minimum, one row per image with:
    - a ground-truth class column   (any of: true_class, label, ground_truth, class)
    - a predicted class column      (any of: predicted_class, pred_class, prediction)
    - a sample identifier column    (any of: filename, sample_id, image_path, image_id)
      -> used only for traceability/logging, not required for the metric math.

Column names are matched case-insensitively against the aliases above, so the
script does not break if upstream naming changes slightly.

Outputs
-------
    outputs/accuracy_delta.csv
        columns: class, method, accuracy_clean, accuracy_modified, delta,
                 f1_clean, f1_modified
        (delta = accuracy_clean - accuracy_modified, on a 0-1 scale)

    outputs/accuracy_delta.png
        Grouped bar chart: one group per class, 3 bars per group
        (blur / replace / crop), bar height = accuracy delta.
        A horizontal reference line is drawn at delta = 0.10 (10%), which is
        the project's threshold for "significant spurious reliance" on that
        modification's feature type (e.g. logos/text).

Usage
-----
    python scripts/accuracy_delta.py

    Optional flags let you point at a different inference directory or
    output directory, e.g.:

    python scripts/accuracy_delta.py \
        --inference-dir outputs/inference \
        --output-dir outputs
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# Column-name aliases so the script is tolerant of minor naming drift
# between teammates' CSVs.
# ----------------------------------------------------------------------
TRUE_LABEL_ALIASES = ["true_class", "label", "ground_truth", "class", "true_label"]
PRED_LABEL_ALIASES = ["predicted_class", "pred_class", "prediction", "pred"]
ID_ALIASES = ["filename", "sample_id", "image_path", "image_id"]

METHODS = ["blur", "replace", "crop"]
SIGNIFICANCE_THRESHOLD = 0.10  # 10%, per the task spec


def _find_column(df: pd.DataFrame, aliases: list[str], required_for: str) -> str:
    """Return the first matching column name (case-insensitive) from aliases."""
    lower_map = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    raise ValueError(
        f"Could not find a column for '{required_for}'. "
        f"Looked for any of {aliases} in columns {list(df.columns)}"
    )


def load_inference_csv(path: Path) -> pd.DataFrame:
    """Load one inference CSV and normalise its columns to:
    ['id', 'true_class', 'predicted_class']
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}\n"
            f"(Make sure run_inference.py has been run for all four "
            f"variants: clean, blur, replace, crop.)"
        )

    df = pd.read_csv(path)

    true_col = _find_column(df, TRUE_LABEL_ALIASES, "ground-truth label")
    pred_col = _find_column(df, PRED_LABEL_ALIASES, "predicted label")

    try:
        id_col = _find_column(df, ID_ALIASES, "sample id")
    except ValueError:
        # Not strictly required for the metrics, fall back to row index.
        df["_row_id"] = df.index.astype(str)
        id_col = "_row_id"

    out = df[[id_col, true_col, pred_col]].copy()
    out.columns = ["id", "true_class", "predicted_class"]

    # Normalise label text (strip whitespace, keep original case otherwise
    # to avoid silently merging genuinely different classes).
    out["true_class"] = out["true_class"].astype(str).str.strip()
    out["predicted_class"] = out["predicted_class"].astype(str).str.strip()

    return out


def per_class_accuracy(df: pd.DataFrame) -> dict:
    """accuracy = correct predictions / total predictions, computed per class
    (class = ground-truth class)."""
    result = {}
    for cls, group in df.groupby("true_class"):
        correct = (group["predicted_class"] == group["true_class"]).sum()
        total = len(group)
        result[cls] = correct / total if total > 0 else float("nan")
    return result


def per_class_f1(df: pd.DataFrame) -> dict:
    """One-vs-rest precision/recall/F1 per class, computed over the whole
    file (predictions for ALL classes), which is the standard way F1 is
    defined for a multi-class classifier.
    """
    classes = sorted(set(df["true_class"]) | set(df["predicted_class"]))
    result = {}
    for cls in classes:
        tp = ((df["predicted_class"] == cls) & (df["true_class"] == cls)).sum()
        fp = ((df["predicted_class"] == cls) & (df["true_class"] != cls)).sum()
        fn = ((df["predicted_class"] != cls) & (df["true_class"] == cls)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        result[cls] = f1
    return result


def build_results_table(clean_df: pd.DataFrame, modified_dfs: dict) -> pd.DataFrame:
    """
    Build the long-format results table:
    class, method, accuracy_clean, accuracy_modified, delta, f1_clean, f1_modified
    """
    acc_clean = per_class_accuracy(clean_df)
    f1_clean = per_class_f1(clean_df)

    rows = []
    all_classes = sorted(acc_clean.keys())

    for method in METHODS:
        mod_df = modified_dfs[method]
        acc_mod = per_class_accuracy(mod_df)
        f1_mod = per_class_f1(mod_df)

        for cls in all_classes:
            a_clean = acc_clean.get(cls, float("nan"))
            a_mod = acc_mod.get(cls, float("nan"))
            delta = (
                a_clean - a_mod
                if not (pd.isna(a_clean) or pd.isna(a_mod))
                else float("nan")
            )
            rows.append(
                {
                    "class": cls,
                    "method": method,
                    "accuracy_clean": round(a_clean, 4) if not pd.isna(a_clean) else a_clean,
                    "accuracy_modified": round(a_mod, 4) if not pd.isna(a_mod) else a_mod,
                    "delta": round(delta, 4) if not pd.isna(delta) else delta,
                    "f1_clean": round(f1_clean.get(cls, float("nan")), 4),
                    "f1_modified": round(f1_mod.get(cls, float("nan")), 4),
                }
            )

    return pd.DataFrame(rows)


def plot_grouped_bar(results: pd.DataFrame, output_path: Path) -> None:
    """Grouped bar chart: 3 bars (blur/replace/crop) per class, bar = delta."""
    classes = sorted(results["class"].unique())
    x = np.arange(len(classes))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.2), 6))

    colors = {"blur": "#4C72B0", "replace": "#DD8452", "crop": "#55A868"}

    for i, method in enumerate(METHODS):
        sub = results[results["method"] == method].set_index("class")
        deltas = [sub.loc[cls, "delta"] if cls in sub.index else float("nan") for cls in classes]
        offset = (i - 1) * bar_width
        ax.bar(
            x + offset,
            deltas,
            width=bar_width,
            label=method,
            color=colors.get(method),
        )

    ax.axhline(
        SIGNIFICANCE_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"{int(SIGNIFICANCE_THRESHOLD * 100)}% significance threshold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Accuracy delta (clean − modified)")
    ax.set_title("Accuracy Drop by Class and Modification Method\n(higher = more spurious reliance on that feature)")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute accuracy/F1 delta between clean and modified images.")
    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=Path("outputs/inference"),
        help="Directory containing inference_clean.csv, inference_blur.csv, "
        "inference_replace.csv, inference_crop.csv (default: outputs/inference)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/metrics"),
        help="Directory to write accuracy_delta.csv / .png into (default: outputs/metrics, "
        "matching where src/dashboard/callbacks.py reads it from)",
    )
    args = parser.parse_args()

    inference_dir = args.inference_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        clean_df = load_inference_csv(inference_dir / "inference_clean.csv")
        modified_dfs = {
            "blur": load_inference_csv(inference_dir / "inference_blur.csv"),
            "replace": load_inference_csv(inference_dir / "inference_replace.csv"),
            "crop": load_inference_csv(inference_dir / "inference_crop.csv"),
        }
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    results = build_results_table(clean_df, modified_dfs)

    csv_path = output_dir / "accuracy_delta.csv"
    results.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(results)} rows)")

    png_path = output_dir / "accuracy_delta.png"
    plot_grouped_bar(results, png_path)
    print(f"Wrote {png_path}")

    # Console summary of any significant spurious-reliance findings.
    flagged = results[results["delta"] > SIGNIFICANCE_THRESHOLD]
    if len(flagged) > 0:
        print("\nSignificant spurious reliance detected (delta > 10%):")
        for _, row in flagged.iterrows():
            print(
                f"  - class={row['class']!r}, method={row['method']}, "
                f"delta={row['delta']:.2%}"
            )
    else:
        print("\nNo class/method combination exceeded the 10% significance threshold.")


if __name__ == "__main__":
    main()
