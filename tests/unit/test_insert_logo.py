"""
Unit test for overlay_logo — checks the bounding box math and that the
image is actually modified, without needing real image files.

Run: pytest tests/unit/test_insert_logo.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_prep.insert_logo import overlay_logo


@pytest.fixture
def base_image():
    return np.full((224, 224, 3), 128, dtype=np.uint8)


@pytest.fixture
def logo_with_alpha():
    logo = np.zeros((50, 50, 4), dtype=np.uint8)
    logo[:, :, :3] = 255  # white logo
    logo[:, :, 3] = 255   # fully opaque
    return logo


def test_bbox_is_within_image_bounds(base_image, logo_with_alpha):
    cfg = {"relative_size": 0.18, "position": "lower_left", "opacity": 0.9}
    _, bbox = overlay_logo(base_image, logo_with_alpha, cfg)

    assert 0 <= bbox["x"] < base_image.shape[1]
    assert 0 <= bbox["y"] < base_image.shape[0]
    assert bbox["x"] + bbox["w"] <= base_image.shape[1]
    assert bbox["y"] + bbox["h"] <= base_image.shape[0]


def test_image_is_actually_modified(base_image, logo_with_alpha):
    cfg = {"relative_size": 0.18, "position": "lower_left", "opacity": 0.9}
    modified, bbox = overlay_logo(base_image, logo_with_alpha, bbox_cfg := cfg)

    region = modified[bbox["y"]:bbox["y"] + bbox["h"], bbox["x"]:bbox["x"] + bbox["w"]]
    # at opacity 0.9 with a white logo over a grey (128) background,
    # the region should be much brighter than the untouched background
    assert region.mean() > base_image.mean()


def test_unmodified_region_outside_bbox(base_image, logo_with_alpha):
    cfg = {"relative_size": 0.18, "position": "lower_left", "opacity": 0.9}
    modified, bbox = overlay_logo(base_image, logo_with_alpha, cfg)

    # top-right corner should be untouched by a lower-left logo
    corner = modified[0:10, -10:]
    assert np.array_equal(corner, base_image[0:10, -10:])


def test_lower_right_position(base_image, logo_with_alpha):
    cfg = {"relative_size": 0.18, "position": "lower_right", "opacity": 0.9}
    _, bbox = overlay_logo(base_image, logo_with_alpha, cfg)
    # logo should be in the right half of the image
    assert bbox["x"] > base_image.shape[1] / 2


def test_invalid_position_raises():
    cfg = {"relative_size": 0.18, "position": "center", "opacity": 0.9}
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    logo = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        overlay_logo(img, logo, cfg)
