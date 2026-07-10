"""
torchvision_resnet_model.py
=============================

Adapter wrapping a plain ImageNet-supervised torchvision ResNet-50 (cross-
entropy trained on labels only, no image-text/caption supervision at all)
so it plugs into the exact same pipeline as `CLIPModel` — `GradCAM`,
`BiLRP`, `LRPExplainer`, and `CHMitigator` all work against this class
completely unmodified, because they're coupled to a *shape*
(`.device`/`.preprocess`/`.model.visual`/`.model.encode_image()`/
`.classifier`), not to `CLIPModel` by type.

This is the "expected to be robust" control model in the Clever-Hans logo
experiment: since it never saw natural-language captions during training,
it has no incentive to treat pasted watermark text as a semantic signal
the way CLIP does (the documented "typographic attack" phenomenon) — a
large clean-vs-logo accuracy gap here would suggest the shortcut isn't
CLIP-specific after all, while a small gap is the expected result.

Notes on the shape match
--------------------------
- `.model.visual` is the torchvision ResNet-50 itself (with `.fc` replaced
  by `nn.Identity()` so it returns pooled 2048-d features, analogous to
  CLIP's 1024-d image embedding) — same Bottleneck/conv/bn/relu structure
  Grad-CAM and LRP already expect, including a top-level `.conv1` attribute
  LRP's `_ClassifierHead` reads directly.
- `CHMitigator`'s default `module_name="relu3"` (CLIP RN50's 3-conv stem)
  doesn't exist here — torchvision's stem is a single conv, so its only
  top-level early relu is named `"relu"` (call
  `CHMitigator(supervised_model, module_name="relu")`).
- Preprocessing uses standard ImageNet normalization stats, NOT CLIP's —
  a different pretraining distribution entirely.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _VisualEncoder(nn.Module):
    """Thin wrapper giving the torchvision ResNet-50 the same `.visual` +
    `.encode_image()` shape as CLIP's raw `model` object, so downstream XAI
    code (written against `clip_model.model.visual` /
    `clip_model.model.encode_image(tensor)`) doesn't need to know which
    backbone it's looking at."""

    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.visual = resnet

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


class SupervisedResNetModel:
    """Adapter matching `CLIPModel`'s public shape, backed by a plain
    ImageNet-supervised torchvision ResNet-50 instead of CLIP.

    Parameters
    ----------
    classifier_path:
        Path to a pickled sklearn `LogisticRegression` fit on THIS model's
        2048-d embeddings — not interchangeable with a classifier fit on
        CLIP's 1024-d embeddings.
    device:
        'cuda' or 'cpu'. Defaults to cuda if available.
    weights:
        Passed to `torchvision.models.resnet50(weights=...)`. Defaults to
        the standard supervised ImageNet-1k weights.
    """

    def __init__(
        self,
        classifier_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        weights: str = "IMAGENET1K_V2",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.architecture = "resnet"

        resnet = models.resnet50(weights=weights)
        resnet.fc = nn.Identity()  # return pooled 2048-d features, not 1000-way logits
        self.model = _VisualEncoder(resnet).to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        self.classifier = None
        if classifier_path is not None:
            self.load_classifier(classifier_path)

    def load_classifier(self, path: Union[str, Path]) -> None:
        with open(path, "rb") as f:
            self.classifier = pickle.load(f)

    @torch.no_grad()
    def encode_image(self, img: Image.Image) -> np.ndarray:
        """Return the 2048-dim embedding for a single PIL image (no grad)."""
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(img_t)
        return feat.cpu().numpy().flatten()

    def encode_image_tensor(self, img: Image.Image, requires_grad: bool = True):
        """Live torch tensor (grad-enabled) version, for Grad-CAM/BiLRP —
        matches `CLIPModel.encode_image_tensor`'s signature."""
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        img_t.requires_grad_(requires_grad)
        return self.model.encode_image(img_t), img_t

    def predict(self, features: np.ndarray):
        if self.classifier is None:
            raise RuntimeError("No classifier loaded. Call load_classifier() first.")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.classifier.predict(features)
