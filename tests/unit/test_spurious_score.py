"""
Unit test for heatmap_energy_ratio — pure function, no model/GPU needed,
runs in milliseconds. This is the kind of test everyone should write for
their own module: small, fast, no external dependencies, checks one piece
of logic against a known-correct answer.

Run: pytest tests/unit/test_spurious_score.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.quant.spurious_score import heatmap_energy_ratio


def test_all_energy_inside_box_gives_ratio_one():
    heatmap = np.zeros((10, 10))
    heatmap[2:5, 2:5] = 1.0  # all relevance inside box
    bbox = {"x": 0, "y": 0, "w": 10, "h": 10}  # box covers whole image
    assert heatmap_energy_ratio(heatmap, bbox) == pytest.approx(1.0)


def test_no_energy_inside_box_gives_ratio_zero():
    heatmap = np.zeros((10, 10))
    heatmap[8:10, 8:10] = 1.0  # relevance in bottom-right corner only
    bbox = {"x": 0, "y": 0, "w": 3, "h": 3}  # box in top-left, no overlap
    assert heatmap_energy_ratio(heatmap, bbox) == pytest.approx(0.0)


def test_half_energy_inside_box_gives_ratio_half():
    heatmap = np.zeros((10, 10))
    heatmap[0:5, :] = 1.0   # top half
    heatmap[5:10, :] = 1.0  # bottom half — equal energy, 200 total
    bbox = {"x": 0, "y": 0, "w": 10, "h": 5}  # box = top half exactly
    assert heatmap_energy_ratio(heatmap, bbox) == pytest.approx(0.5)


def test_zero_total_energy_returns_zero_not_nan():
    """Guards against a real failure mode: an all-zero heatmap (e.g. from a
    misconfigured Grad-CAM target layer) shouldn't produce NaN/division
    errors that silently corrupt downstream aggregation."""
    heatmap = np.zeros((10, 10))
    bbox = {"x": 0, "y": 0, "w": 5, "h": 5}
    result = heatmap_energy_ratio(heatmap, bbox)
    assert result == 0.0
    assert not np.isnan(result)


def test_negative_relevance_is_excluded():
    """Grad-CAM can produce negative relevance; only positive evidence
    should count toward the spurious-reliance score."""
    heatmap = np.full((10, 10), -1.0)
    heatmap[0:3, 0:3] = 5.0  # only positive region
    bbox = {"x": 0, "y": 0, "w": 3, "h": 3}
    assert heatmap_energy_ratio(heatmap, bbox) == pytest.approx(1.0)
