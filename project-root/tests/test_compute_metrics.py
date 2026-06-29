```python
import pandas as pd
from pathlib import Path

from scripts.compute_metrics import compute_metrics


def test_compute_metrics(tmp_path):
    # Create fake inference csv
    df = pd.DataFrame({
        "true_label": [
            "Cat",
            "Cat",
            "Dog",
            "Dog",
            "Bird",
            "Bird"
        ],
        "predicted_label": [
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

    output_dir = tmp_path / "outputs"

    compute_metrics(
        str(input_csv),
        str(output_dir)
    )

    metrics_file = output_dir / "metrics_clean.csv"
    confusion_file = output_dir / "confusion_clean.png"

    assert metrics_file.exists()
    assert confusion_file.exists()

    metrics = pd.read_csv(metrics_file)

    assert "Class" in metrics.columns
    assert "Precision" in metrics.columns
    assert "Recall" in metrics.columns
    assert "F1" in metrics.columns
    assert "Accuracy" in metrics.columns
```
