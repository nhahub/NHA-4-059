"""
ch_mitigation.py
=================

Port of Section 4 of the notebook: identifies which filters in
`model.visual.relu3` react most strongly to logo insertion, then masks the
top-N of them at inference time to test whether that recovers accuracy.

Caveat carried over from the review (kept here rather than buried in a
docstring elsewhere, since anyone re-running this should see it):
`relu3` is a very early layer — part of CLIP's 3-conv ResNet stem, before
the first residual block. It encodes low-level texture/edge statistics,
not semantic "logo" concepts. Masking filters there can recover accuracy
for reasons that aren't necessarily "we removed the logo-detecting
neuron" — it can also just act as noise/regularization. Treat the
recovered-accuracy number as suggestive, not as proof the masked filters
specifically encode "logo," and consider ablating a *different* random
set of relu3 filters (of the same size) as a control to see if the
recovery is specific or just injected noise.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.data.image_modifier import BL_WIDTH_FRAC, LOGO_ASPECT_RATIO, TR_WIDTH_FRAC


class CHMitigator:
    def __init__(self, clip_model, module_name: str = "relu3"):
        """
        Parameters
        ----------
        clip_model:
            An instance of src.models.clip_model.CLIPModel (already loaded).
        module_name:
            Name of the module within `model.visual` to hook and mask.
            Defaults to 'relu3', matching the notebook — see the module
            docstring caveat about what layer this actually is.
        """
        self.clip_model = clip_model
        self.model = clip_model.model
        self.module_name = module_name
        self.filter_indices: Optional[np.ndarray] = None
        self._mask_handle = None

    # ------------------------------------------------------------------
    # Step 1: find which filters react most to the logo being present
    # ------------------------------------------------------------------
    def compute_filter_sensitivity(
        self,
        examples: List[dict],
        decode_image: Callable[[dict], Image.Image],
        apply_logo: Callable[[Image.Image], Image.Image],
        sample_size: int = 200,
    ) -> np.ndarray:
        """Returns filter indices sorted ascending by how much their
        activation changes, on average, between clean and logo-inserted
        versions of the same images (most-sensitive filters last)."""
        hook_output = {}

        def hook_fn(module, inp, output):
            hook_output["act"] = output.detach().cpu()

        target_module = dict(self.model.visual.named_modules())[self.module_name]
        handle = target_module.register_forward_hook(hook_fn)

        act_diffs = []
        n = min(sample_size, len(examples))

        with torch.no_grad():
            for i in tqdm(range(n), desc=f"Computing {self.module_name} filter diffs"):
                example = examples[i]
                img_clean = decode_image(example["image"])
                inp_clean = self.clip_model.preprocess(img_clean).unsqueeze(0).to(self.clip_model.device)
                self.model.encode_image(inp_clean)
                act_clean = hook_output["act"].numpy()

                img_logo = apply_logo(img_clean)
                inp_logo = self.clip_model.preprocess(img_logo).unsqueeze(0).to(self.clip_model.device)
                self.model.encode_image(inp_logo)
                act_logo = hook_output["act"].numpy()

                diff = np.abs(act_logo - act_clean).mean(axis=(0, 2, 3))
                act_diffs.append(diff)

        handle.remove()

        avg_diffs = np.mean(act_diffs, axis=0)
        self.filter_indices = np.argsort(avg_diffs)
        return self.filter_indices

    # ------------------------------------------------------------------
    # Step 2: mask the top-N most logo-sensitive filters at inference time
    # ------------------------------------------------------------------
    def apply_mitigation(self, num_filters: int = 5) -> None:
        """Registers a forward hook on `module_name` that zeroes out the top
        `num_filters` most logo-sensitive channels, in both the bottom-left
        and top-right bands where `ImageModifier.paste_logo` actually draws
        (derived from its `BL_WIDTH_FRAC`/`TR_WIDTH_FRAC`/`LOGO_ASPECT_RATIO`
        rather than a hand-picked row cutoff — the previous fixed
        `y_offset=90` only ever zeroed the bottom band, so it silently
        stopped covering the top-right logo once that box was enlarged).
        Row fractions are computed against the hooked layer's *actual*
        spatial size at call time, so this keeps working if `module_name`
        points at a different-resolution layer."""
        if self.filter_indices is None:
            raise RuntimeError("Call compute_filter_sensitivity() first.")

        top_filters = self.filter_indices[-num_filters:]
        bl_row_frac = BL_WIDTH_FRAC * LOGO_ASPECT_RATIO
        tr_row_frac = TR_WIDTH_FRAC * LOGO_ASPECT_RATIO

        def hook(module, inp, output):
            mask = torch.ones_like(output)
            h = output.shape[-2]
            bottom_from = h - int(h * bl_row_frac)
            top_to = int(h * tr_row_frac)
            for f in top_filters:
                mask[:, f, bottom_from:, :] = 0  # bottom-left logo band
                mask[:, f, :top_to, :] = 0        # top-right logo band
            return output * mask

        target_module = dict(self.model.visual.named_modules())[self.module_name]
        self._mask_handle = target_module.register_forward_hook(hook)

    def remove_mitigation(self) -> None:
        if self._mask_handle is not None:
            self._mask_handle.remove()
            self._mask_handle = None
