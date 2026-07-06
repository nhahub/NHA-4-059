"""
experiment_log.py
===================

Appends one row per run to `outputs/experiment_log.csv`, matching the
SAD §6.2 `experiment_log.csv` schema (exp_id, model_name, dataset_id,
xai_method, accuracy, focus_score, shortcut_detected, heatmap_path,
timestamp, notes). Kept under `outputs/` rather than the SAD's `results/`
to match the output directory this codebase already uses everywhere else
(`outputs/inference`, `outputs/metrics`, `outputs/heatmaps`).
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

FIELDNAMES = [
    "exp_id",
    "model_name",
    "dataset_id",
    "xai_method",
    "accuracy",
    "focus_score",
    "shortcut_detected",
    "heatmap_path",
    "timestamp",
    "notes",
]


class ExperimentLogger:
    def __init__(self, log_path: Union[str, Path] = "outputs/experiment_log.csv"):
        self.log_path = Path(log_path)

    def log(
        self,
        model_name: str,
        dataset_id: str,
        xai_method: str,
        accuracy: float,
        focus_score: float,
        shortcut_detected: bool,
        heatmap_path: str,
        notes: str = "",
    ) -> str:
        """Appends one experiment row and returns its exp_id."""
        timestamp = datetime.now(timezone.utc)
        # microsecond suffix: exp_id is the schema's primary key, and two
        # runs started within the same second would otherwise collide.
        exp_id = f"exp_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

        row = {
            "exp_id": exp_id,
            "model_name": model_name,
            "dataset_id": dataset_id,
            "xai_method": xai_method,
            "accuracy": accuracy,
            "focus_score": focus_score,
            "shortcut_detected": shortcut_detected,
            "heatmap_path": heatmap_path,
            "timestamp": timestamp.isoformat(),
            "notes": notes,
        }

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        return exp_id
