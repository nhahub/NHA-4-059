"""
bilrp.py
========

Pairwise ("BiLRP-style") attribution between two images, showing which
region of image A and which region of image B jointly drive their CLIP
cosine similarity.

THE BUG (Section 5C of CH_Detection_Pipeline.ipynb)
-----------------------------------------------------
The original code computed two *independent* per-image gradient-saliency
maps (`g1`, `g2` — how much each pixel affects the similarity, computed
separately) and then took their outer product:

    pairwise = np.outer(R1.flatten(), R2.flatten())

That means: if patch i in image A is individually "important" (high
gradient magnitude on its own, regardless of image B) and patch j in
image B is also individually "important," the code draws a strong
connecting line between them — with zero information about whether those
two patches actually correspond to, or interact with, each other. A logo
patch and an unrelated high-contrast patch (e.g. a windshield) in the
other image get a strong line for exactly the same reason: both are
locally high-gradient, not because the model is "matching" them. This is
not what real BiLRP (Eberle et al.) computes — real BiLRP propagates
relevance jointly through the similarity computation so a connection
reflects an actual second-order interaction.

THE FIX
-------
Compute the actual mixed second derivative (a Hessian-vector-style joint
attribution) of the similarity with respect to a patch in image A and a
patch in image B:

    R[i, j] = d/d(patch_j in B) [ d(sim)/d(patch_i in A) ]

This is a genuine cross term: it is zero if the similarity's dependence
on patch i (image A) doesn't change depending on patch j (image B), and
non-zero exactly when the model's similarity score depends *jointly* on
the two patches together — which is what a "the model is matching these
two regions" claim actually requires. It costs one extra backward pass
per patch (`patch_grid**2` total), so it's slower than the naive version,
but tractable for a reasonably small patch grid (default 7x7 = 49 passes).

The old outer-product function is kept below as `compute_bilrp_naive`,
clearly labeled, purely for side-by-side comparison / ablation — it should
not be used as evidence on its own.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def _math_sdp_backend():
        return sdpa_kernel(SDPBackend.MATH)
except ImportError:  # older torch without torch.nn.attention.sdpa_kernel
    @contextmanager
    def _math_sdp_backend():
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            yield


def compute_bilrp(
    clip_model,
    img1: Image.Image,
    img2: Image.Image,
    patch_grid: int = 7,
) -> Tuple[np.ndarray, float]:
    """True joint (mixed-second-derivative) pairwise relevance between two
    images' patches, w.r.t. their CLIP cosine similarity.

    Returns
    -------
    pairwise: np.ndarray, shape (patch_grid, patch_grid, patch_grid, patch_grid)
        pairwise[i, j, k, l] = joint relevance between patch (i, j) of img1
        and patch (k, l) of img2. Normalized to [-1, 1] by max abs value.
    sim: float
        The CLIP cosine similarity between the two images.
    """
    device = clip_model.device
    preprocess = clip_model.preprocess
    model = clip_model.model

    img1_t = preprocess(img1).unsqueeze(0).to(device)
    img2_t = preprocess(img2).unsqueeze(0).to(device)
    img1_t.requires_grad_(True)
    img2_t.requires_grad_(True)

    # Force the "math" SDPA backend for the forward pass: CLIP RN50's
    # attention-pool layer otherwise dispatches to the fused
    # efficient-attention kernel, whose backward has no double-backward
    # implementation — and BiLRP needs one (create_graph=True below, then
    # a second backward through that graph for the cross term).
    with _math_sdp_backend():
        f1 = model.encode_image(img1_t)
        f2 = model.encode_image(img2_t)
    f1n = f1 / f1.norm(dim=-1, keepdim=True)
    f2n = f2 / f2.norm(dim=-1, keepdim=True)
    sim = (f1n * f2n).sum()

    # First backward: gradient of sim w.r.t. image1 pixels, kept differentiable
    # (create_graph=True) so we can backprop through *it* w.r.t. image2 next.
    g1 = torch.autograd.grad(sim, img1_t, create_graph=True)[0]  # (1, 3, H, W)
    g1_energy = g1.abs().sum(dim=1, keepdim=True)  # (1, 1, H, W)

    _, _, H, W = g1_energy.shape
    ph = pw = patch_grid
    sh, sw = H // ph, W // pw

    pairwise = np.zeros((ph, pw, ph, pw), dtype=np.float32)

    for i in range(ph):
        for j in range(pw):
            patch_energy = g1_energy[0, 0, i * sh:(i + 1) * sh, j * sw:(j + 1) * sw].sum()

            # Mixed partial: how does this patch's influence-on-similarity
            # (via img1) itself depend on img2's pixels?
            (grad2,) = torch.autograd.grad(patch_energy, img2_t, retain_graph=True)
            grad2_energy = grad2.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()  # (H, W)

            for k in range(ph):
                for l in range(pw):
                    pairwise[i, j, k, l] = grad2_energy[k * sh:(k + 1) * sh, l * sw:(l + 1) * sw].sum()

    max_abs = np.abs(pairwise).max()
    if max_abs > 0:
        pairwise = pairwise / max_abs

    return pairwise, float(sim.item())


def compute_bilrp_naive(
    clip_model,
    img1: Image.Image,
    img2: Image.Image,
    stride: int = 16,
) -> Tuple[np.ndarray, float]:
    """The ORIGINAL notebook's method: outer product of two independent
    per-image gradient-saliency maps. Kept only for comparison — see the
    module docstring for why this doesn't demonstrate cross-image
    correspondence. Do not present this on its own as evidence of the
    model 'matching' two regions.
    """
    device = clip_model.device
    preprocess = clip_model.preprocess
    model = clip_model.model

    img1_t = preprocess(img1).unsqueeze(0).to(device).requires_grad_(True)
    img2_t = preprocess(img2).unsqueeze(0).to(device).requires_grad_(True)

    f1 = model.encode_image(img1_t)
    f2 = model.encode_image(img2_t)
    f1_n = f1 / f1.norm(dim=-1, keepdim=True)
    f2_n = f2 / f2.norm(dim=-1, keepdim=True)
    sim = (f1_n * f2_n).sum()

    model.zero_grad()
    sim.backward()

    g1 = np.abs(img1_t.grad.squeeze().cpu().numpy()).sum(axis=0)
    g2 = np.abs(img2_t.grad.squeeze().cpu().numpy()).sum(axis=0)

    h, w = g1.shape
    ph, pw = h // stride, w // stride
    R1, R2 = np.zeros((ph, pw)), np.zeros((ph, pw))
    for i in range(ph):
        for j in range(pw):
            R1[i, j] = g1[i * stride:(i + 1) * stride, j * stride:(j + 1) * stride].sum()
            R2[i, j] = g2[i * stride:(i + 1) * stride, j * stride:(j + 1) * stride].sum()

    R1 = R1 / (R1.max() + 1e-8)
    R2 = R2 / (R2.max() + 1e-8)
    pairwise = np.outer(R1.flatten(), R2.flatten()).reshape(ph, pw, ph, pw)
    return pairwise, sim.item()


def plot_bilrp(
    img1: Image.Image,
    img2: Image.Image,
    pw_R: np.ndarray,
    sim: float,
    title: str = "",
    fname: Optional[str] = None,
    top_k: int = 40,
    show: bool = False,
):
    """Renders the two images side by side with curved lines connecting the
    top-k strongest patch-pair relevances (red = positive, blue = negative).
    Unchanged from the notebook's rendering logic — only the underlying
    `pw_R` computation changed."""
    img1_np = np.array(img1.resize((224, 224))) / 255.0
    img2_np = np.array(img2.resize((224, 224))) / 255.0

    h, w = 224, 224
    wgap = 30
    ph = pw_R.shape[0]
    stride = h // ph

    canvas = np.ones((h, w * 2 + wgap, 3))
    canvas[:, :w, :] = img1_np
    canvas[:, w + wgap:, :] = img2_np

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(canvas)

    flat_R = pw_R.flatten()
    indices = np.argsort(np.abs(flat_R))[-top_k:]

    for idx in indices:
        i, j, k, l = np.unravel_index(idx, pw_R.shape)
        val = flat_R[idx]
        if abs(val) < 0.01:
            continue

        x1 = j * stride + stride // 2
        y1 = i * stride + stride // 2
        x2 = l * stride + stride // 2 + w + wgap
        y2 = k * stride + stride // 2

        alpha = min(1.0, abs(val) ** 0.5 * 2)
        color = "red" if val > 0 else "blue"

        t = np.linspace(0, 1, 30)
        mid_y = 0.5 * (y1 + y2) - 30
        cx = x1 * (1 - t) + x2 * t
        cy = y1 * (1 - t) ** 2 + mid_y * 2 * t * (1 - t) + y2 * t ** 2
        ax.plot(cx, cy, color=color, alpha=alpha, linewidth=0.8)

    ax.set_title(f"{title}\nCLIP Cosine Similarity: {sim:.4f}", fontsize=13, fontweight="bold")
    ax.axis("off")
    if fname:
        plt.savefig(fname, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
