import pytest

pytest.importorskip("pandas")

from scripts.build_dataset_registry import build_registry_row, build_split_csvs


SPLITS = {
    "garbage_truck": {"train": ["a.jpg", "b.jpg"], "val": ["c.jpg"], "test": ["d.jpg"]},
    "pickup": {"train": ["e.jpg"], "val": [], "test": ["f.jpg"]},
}
LOGO_BOXES = {"a.jpg": [1, 1, 2, 2], "f.jpg": [1, 1, 2, 2]}


def test_build_registry_row_counts():
    row = build_registry_row("imagenet_trucks", "Test", SPLITS, "224x224", "")

    assert row["num_classes"] == 2
    assert row["num_samples"] == 5  # 2+1+1 (garbage_truck) + 1+0+1 (pickup)


def test_build_split_csvs_schema_and_has_shortcut_flag():
    dfs = build_split_csvs("imagenet_trucks", SPLITS, LOGO_BOXES)

    assert set(dfs.keys()) == {"train", "val", "test"}

    train_df = dfs["train"]
    expected_cols = {
        "sample_id", "image_path", "dataset_id", "label",
        "label_id", "split", "has_shortcut", "notes",
    }
    assert expected_cols.issubset(train_df.columns)
    assert len(train_df) == 3  # a.jpg, b.jpg, e.jpg

    a_row = train_df[train_df["sample_id"] == "garbage_truck/a.jpg"].iloc[0]
    assert bool(a_row["has_shortcut"]) is True

    b_row = train_df[train_df["sample_id"] == "garbage_truck/b.jpg"].iloc[0]
    assert bool(b_row["has_shortcut"]) is False

    # label_id should be a stable 0..1 index by sorted class name
    assert set(train_df["label_id"].unique()) == {0, 1}
