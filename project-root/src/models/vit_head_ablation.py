"""
vit_head_ablation.py
======================

CH-mitigation analog for CLIP's ViT visual transformer: ranks `(layer,
head)` attention-head pairs by how much their attention pattern changes
between clean and logo-inserted images, then "masks" (zeroes) the top-N
most logo-sensitive heads at inference time — the same clean-vs-logo
sensitivity-ranking idea as `src/models/ch_mitigation.py::CHMitigator`,
adapted to a transformer instead of a conv stem.

Why this is a separate class, not a CHMitigator branch
---------------------------------------------------------
`CHMitigator` hooks a conv layer's *output* and zeroes channels with a
forward hook — straightforward because `nn.Conv2d` output is just a tensor
you can multiply a mask into. There's no equivalent for a single attention
head: `nn.MultiheadAttention` fuses all heads into one call with no
supported API to "zero head 3 only." Ablation here instead replaces the
affected block's `attn.forward` with `manual_mha_forward` (see
`src/xai/vit_attention_utils.py`), which recomputes attention from public,
stable attributes and zeroes the selected heads before the output
projection mixes them back together.

That reimplementation is verified against real `nn.MultiheadAttention` by a
numeric parity test (`tests/test_vit_attention_utils.py`) — treat that test
as a hard prerequisite, not optional, before trusting any result from this
class: a subtly wrong reshape there would silently produce meaningless
ablation results here.

Caveat (same spirit as CHMitigator's relu3 caveat)
-----------------------------------------------------
Ranking by raw attention-weight diff and ablating those heads can recover
accuracy for reasons that aren't necessarily "we removed the logo-attending
head" — it can also just be noise/regularization from disabling any
handful of heads. Treat recovered accuracy as suggestive, and consider
ablating a random same-size set of heads as a control.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.xai.vit_attention_utils import (
    capture_attention_weights,
    manual_mha_forward,
    num_layers_and_heads,
)


class ViTHeadAblationMitigator:
    def __init__(self, clip_model):
        """
        Parameters
        ----------
        clip_model:
            A `src.models.clip_model.CLIPModel` loaded with a ViT `model_name`
            (e.g. `"ViT-B/32"`).
        """
        visual = clip_model.model.visual
        if not hasattr(visual, "transformer"):
            raise TypeError(
                "ViTHeadAblationMitigator requires a ViT-style visual backbone "
                "(model.visual.transformer.resblocks) — got a conv backbone. "
                "Use CHMitigator instead for RN50-style models."
            )
        self.clip_model = clip_model
        self.visual = visual
        self.num_layers, self.num_heads = num_layers_and_heads(visual)
        self.ranked_layer_head_pairs: Optional[List[Tuple[int, int]]] = None
        self._patch_handles: List[Tuple[object, object]] = []

    # ------------------------------------------------------------------
    # Step 1: find which (layer, head) pairs react most to the logo
    # ------------------------------------------------------------------
    def _capture_per_head_weights(self, img: Image.Image) -> List[torch.Tensor]:
        """(num_layers)-length list of (num_heads, L, L) attention weight
        tensors for one image, batch dim stripped."""
        img_t = self.clip_model.preprocess(img).unsqueeze(0).to(self.clip_model.device)
        with torch.no_grad(), capture_attention_weights(self.visual, average=False) as captured:
            self.visual(img_t.type(self.visual.conv1.weight.dtype))
        return [captured[i][0] for i in range(self.num_layers)]

    def compute_head_sensitivity(
        self,
        examples: List[dict],
        decode_image: Callable[[dict], Image.Image],
        apply_logo: Callable[[Image.Image], Image.Image],
        sample_size: int = 200,
    ) -> List[Tuple[int, int]]:
        """Returns `(layer, head)` pairs sorted ascending by how much their
        attention pattern changes, on average, between clean and
        logo-inserted versions of the same images (most-sensitive pairs
        last) — the ViT analog of `CHMitigator.compute_filter_sensitivity`'s
        `filter_indices`."""
        n = min(sample_size, len(examples))
        diffs = np.zeros((self.num_layers, self.num_heads))

        for i in tqdm(range(n), desc="Computing per-head attention diffs"):
            example = examples[i]
            img_clean = decode_image(example["image"])
            img_logo = apply_logo(img_clean)

            weights_clean = self._capture_per_head_weights(img_clean)
            weights_logo = self._capture_per_head_weights(img_logo)

            for layer_idx in range(self.num_layers):
                diff = (weights_logo[layer_idx] - weights_clean[layer_idx]).abs()
                diffs[layer_idx] += diff.mean(dim=(-2, -1)).cpu().numpy()

        diffs /= n
        flat = [(l, h, diffs[l, h]) for l in range(self.num_layers) for h in range(self.num_heads)]
        flat.sort(key=lambda t: t[2])
        self.ranked_layer_head_pairs = [(l, h) for l, h, _ in flat]
        return self.ranked_layer_head_pairs

    # ------------------------------------------------------------------
    # Step 2: ablate the top-N most logo-sensitive heads at inference time
    # ------------------------------------------------------------------
    def apply_mitigation(self, num_heads_to_ablate: int = 5) -> None:
        """Patches the top `num_heads_to_ablate` most logo-sensitive
        `(layer, head)` pairs' `attn.forward` (grouped by layer) to zero
        those heads' contribution, via `manual_mha_forward`. Call
        `remove_mitigation()` before reusing `clip_model.model` for
        anything else — same lifecycle contract as
        `CHMitigator.apply_mitigation`/`remove_mitigation`."""
        if self.ranked_layer_head_pairs is None:
            raise RuntimeError("Call compute_head_sensitivity() first.")

        top_pairs = self.ranked_layer_head_pairs[-num_heads_to_ablate:]
        ablate_by_layer = defaultdict(list)
        for layer_idx, head_idx in top_pairs:
            ablate_by_layer[layer_idx].append(head_idx)

        blocks = self.visual.transformer.resblocks
        self._patch_handles = []
        for layer_idx, heads in ablate_by_layer.items():
            attn = blocks[layer_idx].attn
            orig_forward = attn.forward

            def make_patched(attn=attn, heads=heads):
                def patched(query, key, value, **kwargs):
                    need_weights = kwargs.get("need_weights", False)
                    out, weights = manual_mha_forward(
                        attn, query, ablate_heads=heads, need_weights=need_weights
                    )
                    if need_weights and weights is not None and kwargs.get("average_attn_weights", True):
                        weights = weights.mean(dim=1)
                    return out, weights

                return patched

            attn.forward = make_patched()
            self._patch_handles.append((attn, orig_forward))

    def remove_mitigation(self) -> None:
        for attn, orig_forward in self._patch_handles:
            attn.forward = orig_forward
        self._patch_handles = []
