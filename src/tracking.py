"""
Thin MLflow wrapper so every module logs runs the same way, without each
person reinventing tracking calls. Import this, don't call mlflow directly,
so the experiment name / tracking URI stays consistent project-wide.

Owner: Person F (Integration) sets this up Day 1-2; everyone else just
imports `start_run` and `log_*` helpers from here.

Local tracking only (no remote server needed for a 10-day project) — runs
are stored in ./mlruns and viewable with:
    mlflow ui --backend-store-uri ./mlruns
"""
from contextlib import contextmanager
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = f"file://{ROOT}/mlruns"
EXPERIMENT_NAME = "clever-hans-clip"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


@contextmanager
def start_run(run_name: str, tags: dict = None):
    """
    Usage:
        with start_run("clip_inference_original", tags={"owner": "A", "dataset": "original"}):
            log_params({"backbone": "ViT-B/32"})
            ... do work ...
            log_metrics({"accuracy": 0.83})
            log_artifact_dir("outputs/predictions")
    """
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def log_params(params: dict):
    mlflow.log_params(params)


def log_metrics(metrics: dict, step: int = None):
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str):
    mlflow.log_artifact(path)


def log_artifact_dir(dir_path: str):
    mlflow.log_artifacts(dir_path)


def log_config_snapshot(config: dict):
    """Log the config.yaml contents used for this run, so every run is
    traceable back to the exact settings (backbone, logo params, etc.)
    that produced it — important since config.yaml changes over the
    10 days (e.g. backbone switching from ViT to RN50 on Day 1)."""
    flat = {}
    for section, values in config.items():
        if isinstance(values, dict):
            for k, v in values.items():
                if isinstance(v, (str, int, float, bool)):
                    flat[f"{section}.{k}"] = v
    mlflow.log_params(flat)
