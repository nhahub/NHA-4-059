```python
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


def compute_metrics(input_file, output_dir="outputs"):
    # Read inference results
    df = pd.read_csv(input_file)

    # Check required columns
    required_columns = ["true_label", "predicted_label"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    y_true = df["true_label"]
    y_pred = df["predicted_label"]

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
        "Class": labels,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Support": support
    })

    metrics_df["Accuracy"] = accuracy

    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, "metrics_clean.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

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

    confusion_path = os.path.join(output_dir, "confusion_clean.png")
    plt.tight_layout()
    plt.savefig(confusion_path)
    plt.close()

    print(f"Metrics saved to {metrics_path}")
    print(f"Confusion matrix saved to {confusion_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to inference CSV"
    )

    parser.add_argument(
        "--output_dir",
        default="outputs"
    )

    args = parser.parse_args()

    compute_metrics(args.input, args.output_dir)
```
