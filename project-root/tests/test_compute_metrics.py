import pandas as pd
from pathlib import Path

from scripts.compute_metrics import compute_metrics


def test_compute_metrics(tmp_path):
    # Create fake inference csv, using the columns run_inference.py actually writes
    df = pd.DataFrame({
        "true_class": [
            "Cat",
            "Cat",
            "Dog",
            "Dog",
            "Bird",
            "Bird"
        ],
        "predicted_class": [
            "Cat",
            "Dog",
            "Dog",
            "Dog",
            "Bird",
            "Cat"
        ]
    })

    input_csv = tmp_path / "inference_clean.csv"
    df.to_csv(input_csv, index=False)

    output_dir = tmp_path / "outputs" / "metrics"

    compute_metrics(
        str(input_csv),
        str(output_dir),
        tag="clean",
    )

    metrics_file = output_dir / "metrics_clean.csv"
    confusion_csv = output_dir / "confusion_matrix.csv"
    confusion_png = output_dir / "confusion_clean.png"
    summary_file = output_dir / "summary_metrics.csv"

    assert metrics_file.exists()
    assert confusion_csv.exists()
    assert confusion_png.exists()
    assert summary_file.exists()

    metrics = pd.read_csv(metrics_file)

    assert "class" in metrics.columns
    assert "precision" in metrics.columns
    assert "recall" in metrics.columns
    assert "f1" in metrics.columns
    assert "accuracy" in metrics.columns

    summary = pd.read_csv(summary_file)
    lookup = dict(zip(summary["metric"], summary["value"]))
    assert "overall_accuracy" in lookup
    assert "macro_f1" in lookup
    assert int(lookup["total_images"]) == len(df)
    assert int(lookup["num_classes"]) == 3
