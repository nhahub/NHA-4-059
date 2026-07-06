"""
clip_model.py
=============

Wraps OpenAI CLIP (RN50) plus the LogisticRegression classifier trained on
top of its image embeddings, so the rest of the codebase (CH mitigation,
Grad-CAM, inference scripts) has a single object to import instead of a
pile of loose globals like the original notebook.

This is a straight port of Cell 0 / Cell 2 of CH_Detection_Pipeline.ipynb,
reorganised into a class. No behavioural changes were made here — the bugs
that needed fixing were in XAI code (Section 5), not in the model loading
or classification path.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

try:
    import clip  # openai/CLIP
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "The 'clip' package is required. Install with:\n"
        "  pip install git+https://github.com/openai/CLIP.git"
    ) from e


# ImageNet class-index -> truck-subset label name, as used throughout the
# project (Section 0 of the notebook).
TRUCK_CLASSES: Dict[str, int] = {
    "minivan": 656,
    "moving_van": 675,
    "police_van": 734,
    "fire_engine": 555,
    "garbage_truck": 569,
    "pickup": 717,
    "tow_truck": 864,
    "trailer_truck": 867,
}


def decode_image(img: Union[dict, Image.Image]) -> Image.Image:
    """Decode a HF-datasets-style {'bytes': ...} record or a PIL Image into RGB."""
    if isinstance(img, dict):
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    return img.convert("RGB")


class CLIPModel:
    """Thin, stateful wrapper around CLIP RN50 + a linear classifier head.

    Parameters
    ----------
    classifier_path:
        Path to a pickled sklearn ``LogisticRegression`` (e.g. ``clf_full.pkl``)
        fit on CLIP image embeddings.
    device:
        'cuda' or 'cpu'. Defaults to cuda if available.
    """

    def __init__(self, classifier_path: Optional[Union[str, Path]] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("RN50", device=self.device)
        # clip.load keeps the model in fp16 on CUDA (weights are cast in
        # build_model before .to(device)). Forward-only inference tolerates
        # that fine, but Grad-CAM's backward pass and BiLRP's double backward
        # (create_graph=True) underflow/NaN in fp16 — force fp32 so XAI
        # gradients are numerically stable.
        self.model = self.model.float()
        self.model.eval()

        self.classifier = None
        self.label_to_name = {v: k for k, v in TRUCK_CLASSES.items()}

        if classifier_path is not None:
            self.load_classifier(classifier_path)

    def load_classifier(self, path: Union[str, Path]) -> None:
        with open(path, "rb") as f:
            self.classifier = pickle.load(f)

    @torch.no_grad()
    def encode_image(self, img: Image.Image) -> np.ndarray:
        """Return the 1024-dim CLIP embedding for a single PIL image (no grad)."""
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(img_t)
        return feat.cpu().numpy().flatten()

    def encode_image_tensor(self, img: Image.Image, requires_grad: bool = True) -> torch.Tensor:
        """Return the CLIP embedding as a live torch tensor (grad-enabled),
        for use by XAI code (Grad-CAM, BiLRP) that needs to backprop through it.
        """
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        img_t.requires_grad_(requires_grad)
        return self.model.encode_image(img_t), img_t

    def predict(self, features: np.ndarray):
        """Predict class index(es) from CLIP feature vector(s) using the
        trained LogisticRegression classifier."""
        if self.classifier is None:
            raise RuntimeError("No classifier loaded. Call load_classifier() first.")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.classifier.predict(features)

    def classifier_weight_for_class(self, class_index_in_clf: int) -> torch.Tensor:
        """Return the trained classifier's weight vector (and implicitly the
        bias) for a given class, as a torch tensor on the right device.

        This is the piece Grad-CAM needs to target the *actual* decision
        function instead of an arbitrary embedding dimension. See
        src/xai/gradcam.py for how it's used.
        """
        if self.classifier is None:
            raise RuntimeError("No classifier loaded. Call load_classifier() first.")
        coef = self.classifier.coef_[class_index_in_clf]
        return torch.tensor(coef, dtype=torch.float32, device=self.device)

    def classifier_bias_for_class(self, class_index_in_clf: int) -> float:
        if self.classifier is None:
            raise RuntimeError("No classifier loaded. Call load_classifier() first.")
        return float(self.classifier.intercept_[class_index_in_clf])
