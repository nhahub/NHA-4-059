import pytest

pytest.importorskip("dash")
pytest.importorskip("dash_bootstrap_components")

from src.dashboard.data import RN50_IMAGES, available_images, load_comparison_table
from src.dashboard.layout import serve_layout


def test_serve_layout_builds_without_error():
    """Smoke test: the layout must render regardless of whether results/
    actually has any files yet (fresh checkout, or before the notebook's
    been run) -- it should degrade to an informational message, not crash."""
    layout = serve_layout()
    assert layout is not None


def test_available_images_filters_to_existing_files(tmp_path, monkeypatch):
    import src.dashboard.data as data_module

    monkeypatch.setattr(data_module, "RESULTS_DIR", tmp_path)
    (tmp_path / RN50_IMAGES[0][0]).write_bytes(b"fake png bytes")

    result = available_images(RN50_IMAGES)
    assert result == [RN50_IMAGES[0]]


def test_load_comparison_table_handles_missing_csv(tmp_path, monkeypatch):
    import src.dashboard.data as data_module

    monkeypatch.setattr(data_module, "RESULTS_DIR", tmp_path)
    assert load_comparison_table() is None
