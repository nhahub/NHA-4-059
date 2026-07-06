import json

import pytest
from PIL import Image

from src.data.image_loader import ImageDataLoader


def test_load_single_image_shape(tmp_path):
    """UT-1: load 1 image from disk, verify output tensor shape [3, 224, 224]."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (400, 300), (10, 80, 200)).save(img_path)

    loader = ImageDataLoader()
    tensor = loader.load_single_image(img_path)

    assert tuple(tensor.shape) == (3, 224, 224)


def test_load_single_image_invalid_format_raises_gracefully(tmp_path):
    """UT-2: loading an invalid format (PDF) fails with a clear, catchable
    error instead of an unhandled exception or crash."""
    bad_path = tmp_path / "not_an_image.pdf"
    bad_path.write_bytes(b"%PDF-1.4\nthis is not a real PDF or image\n")

    loader = ImageDataLoader()
    with pytest.raises(ValueError):
        loader.load_single_image(bad_path)


def test_load_images_reads_split_membership_from_splits_json(tmp_path):
    """Loader-level test backing IT-1: load_images() should only return
    images listed for the requested split in splits.json, grouped by the
    dataset_path/{class}/{filename} layout callbacks.py already assumes."""
    splits = {
        "garbage_truck": {"train": ["a.jpg"], "test": ["b.jpg"]},
        "pickup": {"train": ["c.jpg"], "test": ["d.jpg", "missing.jpg"]},
    }
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits))

    dataset_dir = tmp_path / "clean"
    for cls, files in [("garbage_truck", ["a.jpg", "b.jpg"]), ("pickup", ["c.jpg", "d.jpg"])]:
        cls_dir = dataset_dir / cls
        cls_dir.mkdir(parents=True)
        for fname in files:
            Image.new("RGB", (50, 50), (1, 2, 3)).save(cls_dir / fname)

    # Use an identity transform so this test doesn't need torch/torchvision.
    loader = ImageDataLoader(splits_path=splits_path, transform=lambda img: img.size)

    images, labels = loader.load_images(dataset_dir, split="test")

    # 'missing.jpg' is listed in splits.json but doesn't exist on disk —
    # should be skipped, not raise.
    assert len(images) == 2
    assert sorted(labels) == ["garbage_truck", "pickup"]


def test_class_names_and_get_class_name(tmp_path):
    splits = {"pickup": {"test": []}, "garbage_truck": {"test": []}}
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits))

    loader = ImageDataLoader(splits_path=splits_path)

    assert loader.class_names() == ["garbage_truck", "pickup"]
    assert loader.get_class_name(0) == "garbage_truck"
