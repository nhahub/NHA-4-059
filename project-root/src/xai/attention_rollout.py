"""
attention_rollout.py
======================

Attention Rollout (Abnar & Zuidema, 2020) for CLIP's ViT-B/32 visual
transformer — the Grad-CAM analog for a backbone with no conv feature maps
to hook. Produces a (224, 224) relevance map from the CLS token's attention
back to each patch token, composed across all 12 transformer blocks, so it
plugs into the existing `attention_alignment_score(cam, boxes)` from
`gradcam.py` unchanged.

Why not Grad-CAM for ViT
-------------------------
`GradCAM._find_last_conv_name` finds the *last* Conv2d in `model.visual`.
For a ViT, the only Conv2d is `conv1`, the patch-embedding projection at the
very start of the network — hooking it would produce a shallow saliency map
at patch-grid resolution, not a meaningful "where did the classifier's
decision come from" attribution the way it does for a real ResNet's last
residual block. ViT has no equivalent late-stage spatial feature map to
target Grad-CAM at; attention rollout is the standard alternative.

Class-agnostic limitation (accepted, not a bug)
-------------------------------------------------
Plain rollout answers "what did the [CLS] token end up attending to,"
independent of which class the classifier predicted — unlike Grad-CAM,
which explicitly backprops from the predicted class's logit. This is a
known, standard property of the original Abnar & Zuidema formulation, not
something this module tries to work around. A class-aware variant exists
(gradient-weighted rollout, e.g. Chefer et al. 2021, "Transformer
Interpretability Beyond Attention Visualization") but needs a fundamentally
different capture mechanism (per-layer gradients w.r.t. attention weights,
not a plain forward pass) — out of scope here, left as a named future
extension rather than silently bolted on.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image

from src.xai.vit_attention_utils import capture_attention_weights, num_layers_and_heads


def _fuse_residual(attn: torch.Tensor) -> torch.Tensor:
    """Fuse one layer's (row-stochastic) attention matrix with the identity
    (residual/skip connection): `0.5*A + 0.5*I`, row-renormalized. `attn`
    is `(..., L, L)`. The renormalization is a no-op in exact arithmetic
    (a convex combination of two row-stochastic matrices is already
    row-stochastic) but guards against floating-point drift."""
    L = attn.shape[-1]
    eye = torch.eye(L, dtype=attn.dtype, device=attn.device)
    fused = 0.5 * attn + 0.5 * eye
    return fused / fused.sum(dim=-1, keepdim=True)


def rollout(layer_attentions: List[torch.Tensor]) -> torch.Tensor:
    """Compose fused attention matrices across all transformer layers.

    `layer_attentions[i]` is layer `i`'s `(L, L)` attention matrix, where
    `i=0` is the shallowest (first) transformer block. Returns the `(L, L)`
    rolled-out matrix such that `rolled[i, j]` is the total relevance
    flowing from input-side token `j` to output-side token `i`, considering
    every path through every layer.

    Composition order matters and is easy to get backwards without any
    error: layer 0's fused matrix must be applied *first* (i.e. end up
    rightmost/innermost in the matrix product), matching the order tokens
    actually flow through the network. `tests/test_attention_rollout.py`
    has a hand-computed 2-layer/3-token example asserting the exact
    numeric result — getting the order backwards still produces a
    plausible-looking, still-row-stochastic-but-wrong matrix, so a shape
    or range check alone would not catch that bug.
    """
    fused = [_fuse_residual(a) for a in layer_attentions]
    result = fused[0]
    for f in fused[1:]:
        result = f @ result
    return result


def cls_relevance(rolled: torch.Tensor, num_patches: int) -> torch.Tensor:
    """Extract the CLS token's (index 0) relevance over the `num_patches`
    patch tokens (indices 1..num_patches) from a rolled-out `(L, L)`
    matrix, excluding the CLS-CLS self-relevance entry."""
    return rolled[0, 1 : 1 + num_patches]


class AttentionRollout:
    """Grad-CAM-analog explainer for CLIP's ViT visual backbone.

    Parameters
    ----------
    clip_model:
        A `src.models.clip_model.CLIPModel` loaded with `model_name="ViT-B/32"`
        (or any ViT variant — nothing here hardcodes patch count/layer count).
    """

    def __init__(self, clip_model):
        visual = clip_model.model.visual
        if not hasattr(visual, "transformer"):
            raise TypeError(
                "AttentionRollout requires a ViT-style visual backbone "
                "(model.visual.transformer.resblocks) — got a conv backbone. "
                "Use GradCAM instead for RN50-style models."
            )
        self.clip_model = clip_model
        self.visual = visual

    def __call__(self, img: Image.Image) -> np.ndarray:
        """Compute the attention-rollout relevance map for `img`.

        Returns
        -------
        cam: np.ndarray, shape (224, 224), values in [0, 1]
        """
        img_t = self.clip_model.preprocess(img).unsqueeze(0).to(self.clip_model.device)

        with torch.no_grad(), capture_attention_weights(self.visual, average=True) as captured:
            self.visual(img_t.type(self.visual.conv1.weight.dtype))

        num_layers, _ = num_layers_and_heads(self.visual)
        layer_attns = [captured[i][0] for i in range(num_layers)]  # strip batch dim -> (L, L) each

        rolled = rollout(layer_attns)
        num_patches = layer_attns[0].shape[-1] - 1  # -1 for the CLS token
        rel = cls_relevance(rolled, num_patches).cpu().numpy()

        grid = int(round(num_patches ** 0.5))
        cam = rel.reshape(grid, grid)

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))
        return np.array(cam_img) / 255.0
