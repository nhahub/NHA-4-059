"""
test_dashboard_uat.py
=======================

Automates the UAT scenarios (SAD §18 Section 3) that the *current*
dashboard implementation can actually support, using dash's own
browser-driven test harness (`dash.testing`, needs `dash[testing]` +
a chromedriver — install with `pip install dash[testing]` and either a
system Chrome/chromedriver or `webdriver-manager`).

Also covers IT-5 (full pipeline -> dashboard) to the extent it's testable.

IMPORTANT — scope gap found while writing this
------------------------------------------------
`src/dashboard/layout.py`/`components.py` have no `dcc.Upload` component
and `callbacks.py` has no callback that runs live CLIP inference on an
uploaded image. The dashboard as built is a *browse precomputed results*
tool (reads CSVs/PNGs already on disk via dropdowns), not the "upload one
image -> live inference -> heatmap in <8s" tool described in FR-6, UC4,
IT-5, UAT-2, and UAT-3.

That means:
  - UAT-1 (batch analysis results) and UAT-5 (dashboard loads, renders,
    no console errors) ARE testable against what exists, and are
    implemented below.
  - IT-5, UAT-2 (single image upload), UAT-3 (invalid file upload), and
    UAT-4 (model comparison tool — no such UI exists either) are NOT
    automated here because there is nothing to drive: the feature itself
    is missing, not untested. Building it (an upload component + a
    predict-on-upload callback wired to CLIPModel/GradCAM) is a real
    feature gap, not a test-writing gap — flagging rather than faking
    a passing test for functionality that doesn't exist.
"""

import json

import pytest

pytest.importorskip("dash.testing")

import pandas as pd
from PIL import Image


@pytest.fixture
def seeded_project(tmp_path, monkeypatch):
    """Seeds the on-disk layout src/dashboard/callbacks.py reads from
    (outputs/metrics/*.csv, data/clean/{class}/*.jpg) so the dashboard has
    something real to render, then points PROJECT_ROOT-derived paths at it."""
    import src.dashboard.callbacks as callbacks_module

    metrics_dir = tmp_path / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "metric": ["overall_accuracy", "macro_f1", "total_images", "num_classes"],
            "value": [0.87, 0.83, 40, 8],
        }
    ).to_csv(metrics_dir / "summary_metrics.csv", index=False)

    pd.DataFrame(
        {"class": ["garbage_truck", "pickup"], "f1": [0.9, 0.8]}
    ).to_csv(metrics_dir / "metrics_clean.csv", index=False)

    class_dir = tmp_path / "data" / "clean" / "garbage_truck"
    class_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), (10, 80, 200)).save(class_dir / "garbage_truck_0001.jpg")

    monkeypatch.setattr(callbacks_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(callbacks_module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(callbacks_module, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(callbacks_module, "METRICS_DIR", metrics_dir)

    return tmp_path


def _build_app():
    from src.dashboard.main import create_app

    return create_app()


def test_uat5_dashboard_loads_without_console_errors(dash_duo, seeded_project):
    """UAT-5: dashboard loads, all three tabs render, no console errors."""
    app = _build_app()
    dash_duo.start_server(app)

    dash_duo.wait_for_element("#main-tabs", timeout=10)
    assert dash_duo.get_logs() == [] or all(
        "error" not in log["level"].lower() for log in dash_duo.get_logs()
    )

    tab_labels = [el.text for el in dash_duo.find_elements(".nav-link")]
    assert any("SYSTEM PERFORMANCE" in t for t in tab_labels)
    assert any("XAI AUDIT BROWSER" in t for t in tab_labels)
    assert any("SHORTCUT DEPENDENCY" in t for t in tab_labels)


def test_uat1_batch_metrics_render_from_disk(dash_duo, seeded_project):
    """UAT-1 (adapted): the SAD's UAT-1 is "select dataset/model, click Run
    Analysis" — this dashboard has no such trigger; it auto-loads whatever
    batch results already exist on disk on an interval/refresh instead. This
    verifies that precomputed accuracy/F1 KPIs actually reach the page,
    which is the part of UAT-1's acceptance criteria ("metrics displayed")
    this implementation can satisfy."""
    app = _build_app()
    dash_duo.start_server(app)

    dash_duo.wait_for_text_to_equal("#kpi-accuracy", "87.0%", timeout=10)
    dash_duo.wait_for_text_to_equal("#kpi-images", "40", timeout=10)
