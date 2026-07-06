"""
compute_metrics.py
===================

Computes per-class precision/recall/F1/accuracy from an inference CSV
(as written by run_inference.py) and writes the exact files/schema
`src/dashboard/callbacks.py` already reads:

    outputs/metrics/metrics_clean.csv     columns: class, precision, recall,
                                           f1, support, accuracy (lowercase)
    outputs/metrics/confusion_matrix.csv  square matrix, columns = class names
    outputs/metrics/summary_metrics.csv   columns: metric, value
                                           rows: overall_accuracy, macro_f1,
                                           total_images, num_classes
    outputs/metrics/confusion_<tag>.png   seaborn heatmap (human-readable)

Previously this expected `true_label`/`predicted_label` columns and wrote
capitalized column names ("Class", "F1", ...) to a caller-supplied
`outputs/` directory — neither matched what `run_inference.py` produces
(`true_class`/`predicted_class`) or what the dashboard reads
(lowercase columns, under `outputs/metrics/`), so the pipeline broke
between these two stages. Fixed to match both ends.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

TRUE_LABEL_ALIASES = ["true_class", "true_label", "label", "ground_truth"]
PRED_LABEL_ALIASES = ["predicted_class", "predicted_label", "pred_class", "prediction"]


def _find_column(df: pd.DataFrame, aliases, required_for: str) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    raise ValueError(
        f"Could not find a column for '{required_for}'. "
        f"Looked for any of {aliases} in columns {list(df.columns)}"
    )


def compute_metrics(input_file, output_dir="outputs/metrics", tag="clean"):
    # Read inference results
    df = pd.read_csv(input_file)

    true_col = _find_column(df, TRUE_LABEL_ALIASES, "ground-truth label")
    pred_col = _find_column(df, PRED_LABEL_ALIASES, "predicted label")

    y_true = df[true_col].astype(str).str.strip()
    y_pred = df[pred_col].astype(str).str.strip()

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    labels = sorted(y_true.unique())

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0
    )

    metrics_df = pd.DataFrame({
        "class": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support
    })
    metrics_df["accuracy"] = accuracy

    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, f"metrics_{tag}.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Confusion matrix — written as CSV (dashboard reads this directly via
    # px.imshow(df_cm.values, x=cols, y=cols)) and as a PNG for humans.
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_df = pd.DataFrame(cm, columns=labels)
    confusion_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    cm_df.to_csv(confusion_csv_path, index=False)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    confusion_png_path = os.path.join(output_dir, f"confusion_{tag}.png")
    plt.tight_layout()
    plt.savefig(confusion_png_path)
    plt.close()

    # Summary KPIs (dashboard's top-row cards)
    macro_f1 = float(metrics_df["f1"].mean())
    summary_df = pd.DataFrame({
        "metric": ["overall_accuracy", "macro_f1", "total_images", "num_classes"],
        "value": [accuracy, macro_f1, len(df), len(labels)],
    })
    summary_path = os.path.join(output_dir, "summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"Metrics saved to {metrics_path}")
    print(f"Confusion matrix saved to {confusion_csv_path} and {confusion_png_path}")
    print(f"Summary KPIs saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to inference CSV (true_class/predicted_class columns)"
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/metrics"
    )

    parser.add_argument(
        "--tag",
        default="clean",
        help="Suffix used for output filenames, e.g. metrics_<tag>.csv (default: clean)"
    )

    args = parser.parse_args()

    compute_metrics(args.input, args.output_dir, args.tag)
