"""
lrp.py
======

Layer-wise Relevance Propagation for the CLIP-RN50 + LogisticRegression
pipeline, via Zennit (see docs/lrp_setup_notes.md for the feasibility
check this is built on). This is the FR-4/FR-7 "optional" LRP method,
implemented so Grad-CAM and LRP can actually be compared (FR-7).

Why a wrapper module is needed
-------------------------------
Zennit attaches its relevance-propagation rules by walking `model.modules()`
and matching standard `torch.nn` layer types (Conv2d, Linear, BatchNorm2d,
ReLU, AvgPool2d, ...). Two things follow from that:

1. `clip_model.model.visual` alone has no classification head — it ends in
   a 1024-dim embedding. The actual truck-class decision is made by the
   external sklearn `LogisticRegression`, which isn't a torch module at
   all. To get LRP to explain "why did the model call this a garbage
   truck" (the same question Grad-CAM answers here — see gradcam.py), the
   classifier's trained `coef_`/`intercept_` are copied into a real
   `nn.Linear` and appended after the visual encoder, so relevance
   propagates end-to-end from the actual predicted-class logit.

2. CLIP's RN50 visual encoder ends in `AttentionPool2d`, which computes
   its output via a single call to
   `torch.nn.functional.multi_head_attention_forward` — a custom op, not
   a chain of hookable `nn.Module` children. Zennit has no rule for that
   call, so relevance through the attention-pool step falls back to plain
   autograd gradient rather than a true epsilon/z+ LRP rule. Every conv,
   batchnorm, and relu layer *before* attnpool (i.e. essentially the whole
   spatial feature extractor) still gets proper LRP treatment. This is the
   same category of caveat as `ch_mitigation.py`'s relu3 note: treat the
   attention-pool stage as a known approximation, not a limitation that
   invalidates the earlier conv-stack relevance map.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from zennit.attribution import Gradient
from zennit.composites import EpsilonPlusFlat


class _ClassifierHead(nn.Module):
    """visual encoder -> frozen nn.Linear built from the trained
    LogisticRegression's coef_/intercept_, so Zennit sees one continuous
    hookable forward pass ending at the real decision logits."""

    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.model.visual

        clf = clip_model.classifier
        if clf is None:
            raise RuntimeError(
                "LRPExplainer needs a loaded classifier. Call "
                "clip_model.load_classifier(path) first."
            )

        n_classes, n_features = clf.coef_.shape
        self.head = nn.Linear(n_features, n_classes)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor(clf.coef_, dtype=torch.float32))
            self.head.bias.copy_(torch.tensor(clf.intercept_, dtype=torch.float32))
        for p in self.head.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.visual(x.type(self.visual.conv1.weight.dtype))
        return self.head(features.float())


class LRPExplainer:
    def __init__(self, clip_model):
        """
        Parameters
        ----------
        clip_model:
            An instance of src.models.clip_model.CLIPModel with a
            classifier already loaded (`clip_model.classifier`).
        """
        self.clip_model = clip_model
        self.model = _ClassifierHead(clip_model).to(clip_model.device).eval()
        self.composite = EpsilonPlusFlat()

    def __call__(self, img: Image.Image, class_index_in_clf: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Compute the LRP relevance heatmap for `img`.

        Returns
        -------
        heatmap: np.ndarray, shape (224, 224), values in [0, 1]
        predicted_class_index: the classifier-internal class index the
            relevance was computed for.
        """
        img_tensor = self.clip_model.preprocess(img).unsqueeze(0).to(self.clip_model.device)

        with Gradient(model=self.model, composite=self.composite) as attributor:
            if class_index_in_clf is None:
                with torch.no_grad():
                    logits = self.model(img_tensor)
                class_index_in_clf = int(torch.argmax(logits[0]).item())

            n_classes = self.model.head.out_features
            one_hot = torch.eye(n_classes, device=img_tensor.device)[[class_index_in_clf]]

            _, relevance = attributor(img_tensor, one_hot)

        heatmap = relevance.sum(dim=1).squeeze(0).detach().cpu().numpy()  # (H, W)
        heatmap = np.abs(heatmap)

        heatmap = heatmap - heatmap.min()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap, class_index_in_clf


def spatial_correlation(cam_a: np.ndarray, cam_b: np.ndarray) -> float:
    """Pearson correlation between two same-shape attribution maps
    (e.g. Grad-CAM vs LRP), per FR-7. Returns a value in [-1, 1];
    NaN-safe (returns 0.0 if either map is constant)."""
    a = cam_a.flatten().astype(np.float64)
    b = cam_b.flatten().astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
