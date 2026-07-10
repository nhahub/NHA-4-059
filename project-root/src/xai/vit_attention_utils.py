"""
vit_attention_utils.py
=======================

Shared low-level plumbing for getting attention weights out of CLIP's ViT
visual transformer, used by both `attention_rollout.py` (Grad-CAM analog)
and `src/models/vit_head_ablation.py` (CH-mitigation analog). Both need the
same thing CLIP's own code doesn't expose: per-layer, per-head attention
weight matrices.

Why this is needed
-------------------
Each `ResidualAttentionBlock` in `clip_model.model.visual.transformer.resblocks`
holds `self.attn = nn.MultiheadAttention(width, heads)`, called internally as
`self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]` — i.e.
CLIP explicitly discards the attention weights for speed. `attn_mask` is
always `None` for the vision tower (causal masking is a text-transformer-only
concern), so no masking logic is needed anywhere in this module.

Two capture strategies, in order of preference
------------------------------------------------
1. `capture_attention_weights`: monkeypatch each block's bound `attn.forward`
   to force `need_weights=True, average_attn_weights=...` before delegating
   to the real implementation, restoring the original on exit. Cheap, and
   the standard technique public ViT-rollout tools use. The one thing that
   could make this brittle is PyTorch-version differences in
   `nn.MultiheadAttention.forward`'s internal fast-path dispatch.
2. `manual_mha_forward`: a full from-scratch reimplementation of
   multi-head attention using only `nn.MultiheadAttention`'s public,
   documented attributes (`in_proj_weight`/`in_proj_bias`/`out_proj`/
   `num_heads`). Slower (bypasses PyTorch's fused kernels entirely) but
   immune to internal fast-path changes, and the only way to actually
   *ablate* (zero) an individual head's contribution before the output
   projection mixes heads back together — `nn.MultiheadAttention` has no
   supported API for that. Also serves as strategy 1's fallback if
   monkeypatching ever misbehaves.

`manual_mha_forward`'s correctness is verified by a numeric parity test
(`tests/test_vit_attention_utils.py`) comparing its output against real
`nn.MultiheadAttention` with no heads ablated — treat that test as a hard
gate, not a nice-to-have: a subtly wrong reshape here won't crash, it'll
just silently produce wrong attention weights / wrong ablation results.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def manual_mha_forward(
    attn: nn.MultiheadAttention,
    x: torch.Tensor,
    ablate_heads: Optional[Sequence[int]] = None,
    need_weights: bool = False,
):
    """Reimplementation of `nn.MultiheadAttention.forward(x, x, x)` (self-
    attention only — CLIP's vision tower never calls it with different
    query/key/value), using only public/stable attributes so individual
    heads can be zeroed before the output projection mixes them together.

    Parameters
    ----------
    attn:
        A real `nn.MultiheadAttention` module (e.g. a ResidualAttentionBlock's
        `.attn`) to source weights from. Not mutated.
    x:
        Input in CLIP's layout, `(L, N, E)` (batch_first=False).
    ablate_heads:
        Head indices (0-indexed) to zero out entirely (their contribution to
        the output, post-softmax-pre-out_proj), or None to ablate nothing —
        the latter is what the parity test checks against real output.
    need_weights:
        If True, also return the per-head attention weight matrices,
        shape `(N, H, L, L)`.

    Returns
    -------
    (output, weights_or_none)
        output: `(L, N, E)`, matching `nn.MultiheadAttention`'s own output
        shape (ignoring its optional weights return).
    """
    L, N, E = x.shape
    H = attn.num_heads
    head_dim = E // H
    if head_dim * H != E:
        raise ValueError(f"embed_dim {E} not divisible by num_heads {H}")

    W = attn.in_proj_weight  # (3E, E) -- stacked [W_q; W_k; W_v]
    b = attn.in_proj_bias  # (3E,) or None if the module was built with bias=False
    b_q = b[:E] if b is not None else None
    b_k = b[E : 2 * E] if b is not None else None
    b_v = b[2 * E :] if b is not None else None
    q = F.linear(x, W[:E], b_q)
    k = F.linear(x, W[E : 2 * E], b_k)
    v = F.linear(x, W[2 * E :], b_v)

    def split_heads(t: torch.Tensor) -> torch.Tensor:
        # (L, N, E) -> (N*H, L, head_dim)
        return t.reshape(L, N * H, head_dim).transpose(0, 1)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    scale = head_dim ** -0.5
    attn_weights = torch.softmax((q * scale) @ k.transpose(-2, -1), dim=-1)  # (N*H, L, L)
    out = attn_weights @ v  # (N*H, L, head_dim)

    out = out.reshape(N, H, L, head_dim)
    if ablate_heads:
        out = out.clone()
        out[:, list(ablate_heads), :, :] = 0.0
    out = out.transpose(1, 2).reshape(N, L, H * head_dim).transpose(0, 1)  # -> (L, N, E)

    out = F.linear(out, attn.out_proj.weight, attn.out_proj.bias)

    weights_out = attn_weights.reshape(N, H, L, L) if need_weights else None
    return out, weights_out


@contextmanager
def capture_attention_weights(visual_transformer, average: bool = True):
    """Context manager: monkeypatches every ResidualAttentionBlock's
    `attn.forward` in `visual_transformer.transformer.resblocks` to also
    capture attention weights during the next forward pass(es), restoring
    the original `forward` on exit (critical: leaving the patch in place
    would leak into every other explainer sharing the same live
    `clip_model.model` instance).

    Yields a dict `{layer_idx: weights_tensor}`, populated as each block's
    (patched) forward actually runs. `weights_tensor` shape is `(N, L, L)`
    if `average=True` (heads pre-averaged, what Attention Rollout wants) or
    `(N, H, L, L)` if `average=False` (per-head, what head-ablation
    sensitivity ranking wants).

    Falls back to `manual_mha_forward` (strategy 2 in the module docstring)
    if forcing `need_weights=True`/`average_attn_weights=True` on the real
    `nn.MultiheadAttention.forward` raises (e.g. an older/newer PyTorch
    without the `average_attn_weights` kwarg).
    """
    blocks = visual_transformer.transformer.resblocks
    captured: Dict[int, torch.Tensor] = {}
    originals = []

    for layer_idx, block in enumerate(blocks):
        attn = block.attn
        orig_forward = attn.forward

        def make_patched(layer_idx=layer_idx, attn=attn, orig_forward=orig_forward):
            def patched(query, key, value, **kwargs):
                try:
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = average
                    out, weights = orig_forward(query, key, value, **kwargs)
                    if weights is None:
                        raise RuntimeError("real forward returned no weights")
                except (TypeError, RuntimeError):
                    out, weights = manual_mha_forward(
                        attn, query, ablate_heads=None, need_weights=True
                    )
                    if average:
                        weights = weights.mean(dim=1)  # (N,H,L,L) -> (N,L,L)
                captured[layer_idx] = weights.detach()
                return out, weights

            return patched

        attn.forward = make_patched()
        originals.append((attn, orig_forward))

    try:
        yield captured
    finally:
        for attn, orig_forward in originals:
            attn.forward = orig_forward


def num_layers_and_heads(visual_transformer) -> "tuple[int, int]":
    """(num_layers, num_heads) for a CLIP ViT visual transformer — never
    hardcode "12"/"12" elsewhere, derive from the actual model so this keeps
    working if a different ViT variant (e.g. ViT-L) is ever plugged in."""
    blocks = visual_transformer.transformer.resblocks
    return len(blocks), blocks[0].attn.num_heads
