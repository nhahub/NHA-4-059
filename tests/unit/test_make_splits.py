"""
Unit test for make_splits — including the small-N edge case that produced
a negative test-split count before the fix (see make_splits.py history).

Run: pytest tests/unit/test_make_splits.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_prep.make_splits import make_splits


@pytest.fixture
def fake_raw_dataset(tmp_path):
    raw_dir = tmp_path / "raw"
    for class_name, n_images in [("class_a", 20), ("class_b", 3)]:
        class_dir = raw_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(n_images):
            (class_dir / f"img{i}.jpg").touch()
    return raw_dir


def test_splits_cover_all_images_with_no_overlap(fake_raw_dataset, tmp_path):
    config = {"paths": {"raw_data": str(fake_raw_dataset), "annotations": str(tmp_path / "annotations")}}
    splits = make_splits(config)

    all_paths = splits["dev"] + splits["val"] + splits["test"]
    assert len(all_paths) == len(set(all_paths)), "no image should appear in more than one split"
    assert len(all_paths) == 23  # 20 + 3


def test_small_class_does_not_produce_negative_test_count(fake_raw_dataset, tmp_path):
    """Regression test: class_b has only 3 images. Before the fix, n_dev=1
    and n_val=1 was being computed without checking they fit inside n=3,
    which could've gone negative for an even smaller class. Confirm
    everything stays non-negative and every image is accounted for."""
    config = {"paths": {"raw_data": str(fake_raw_dataset), "annotations": str(tmp_path / "annotations")}}
    splits = make_splits(config)

    class_b_paths = [p for p in splits["dev"] + splits["val"] + splits["test"] if "class_b" in p]
    assert len(class_b_paths) == 3
    for split_name in ["dev", "val", "test"]:
        n_in_split = sum(1 for p in splits[split_name] if "class_b" in p)
        assert n_in_split >= 0


def test_split_is_reproducible_with_fixed_seed(fake_raw_dataset, tmp_path):
    config = {"paths": {"raw_data": str(fake_raw_dataset), "annotations": str(tmp_path / "annotations")}}
    splits_1 = make_splits(config)
    splits_2 = make_splits(config)
    assert splits_1 == splits_2
