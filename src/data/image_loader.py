"""
image_loader.py
================

`ImageDataLoader` per SAD §11/§17: loads images for a given split and
returns preprocessed tensors + labels. Built against data that actually
exists in this repo — `data/splits.json` (per-class train/val/test
filename lists, already committed) and a `dataset_path/{class}/{filename}`
folder layout (the same layout `src/dashboard/callbacks.py` already
assumes under `data/clean/`) — rather than a `train/val/test` subfolder
scheme nothing in the project produces.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


def _default_transform(img: Image.Image):
    """CLIP-spec resize/normalise, used only if no transform is supplied.
    Imports torch/torchvision lazily so this module can be imported (and
    ImageDataLoader partially unit-tested) without those installed."""
    import torch
    from torchvision import transforms

    tfm = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    return tfm(img)


class ImageDataLoader:
    """Loads images + labels for a dataset split, using `data/splits.json`
    to decide split membership per class."""

    def __init__(
        self,
        splits_path: Optional[Union[str, Path]] = None,
        transform: Optional[Callable[[Image.Image], object]] = None,
    ):
        self.transform = transform or _default_transform
        self.splits: dict = {}
        if splits_path is not None and Path(splits_path).exists():
            with open(splits_path, "r") as f:
                self.splits = json.load(f)

    def class_names(self) -> List[str]:
        return sorted(self.splits.keys())

    def get_class_name(self, index: int) -> str:
        return self.class_names()[index]

    def load_single_image(self, image_path: Union[str, Path]):
        """Load and preprocess one image. Raises a plain ValueError with a
        clear message on corrupted/unsupported files, instead of letting a
        raw PIL/OS exception propagate (NFR-7: graceful failure)."""
        path = Path(image_path)
        try:
            img = Image.open(path)
            img.load()
            img = img.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Could not load image at {path}: {e}") from e

        return self.transform(img)

    def load_images(self, dataset_path: Union[str, Path], split: str = "test") -> Tuple[list, List[str]]:
        """Loads every image belonging to `split`, across all classes in
        data/splits.json, from `dataset_path/{class}/{filename}`.

        Returns
        -------
        images: list of preprocessed tensors (or whatever `transform` returns)
        labels: list[str], the ground-truth class name per image
        """
        dataset_path = Path(dataset_path)
        images = []
        labels = []

        for class_name, split_map in self.splits.items():
            filenames = split_map.get(split, [])
            for filename in filenames:
                img_path = dataset_path / class_name / filename
                if not img_path.exists():
                    logger.warning("Skipping missing file: %s", img_path)
                    continue
                try:
                    images.append(self.load_single_image(img_path))
                    labels.append(class_name)
                except ValueError as e:
                    logger.warning("Skipping unreadable file: %s", e)

        return images, labels
