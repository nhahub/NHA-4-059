import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from src.xai.vit_attention_utils import manual_mha_forward, num_layers_and_heads


def test_manual_mha_matches_builtin_no_ablation():
    """Mandatory gate (see module docstring): manual_mha_forward must be
    numerically identical to real nn.MultiheadAttention when ablating
    nothing, or every downstream rollout/ablation result built on it is
    meaningless. Covers both the output and the per-head weights."""
    torch.manual_seed(0)
    embed_dim, num_heads, L, N = 8, 2, 5, 3

    attn = nn.MultiheadAttention(embed_dim, num_heads)
    attn.eval()
    x = torch.randn(L, N, embed_dim)

    with torch.no_grad():
        real_out, real_weights = attn(x, x, x, need_weights=True, average_attn_weights=False)
        manual_out, manual_weights = manual_mha_forward(attn, x, ablate_heads=None, need_weights=True)

    assert real_out.shape == manual_out.shape == (L, N, embed_dim)
    assert torch.allclose(real_out, manual_out, atol=1e-4), (
        f"max abs diff: {(real_out - manual_out).abs().max().item()}"
    )

    assert real_weights.shape == manual_weights.shape == (N, num_heads, L, L)
    assert torch.allclose(real_weights, manual_weights, atol=1e-4), (
        f"max abs diff: {(real_weights - manual_weights).abs().max().item()}"
    )


def test_manual_mha_ablate_head_zeros_its_contribution():
    """Ablating a head should change the output (it's not a no-op) but keep
    the shape identical, and ablating ALL heads should zero the entire
    pre-out_proj contribution (out_proj's bias can still make the final
    output nonzero, so we check the pre-bias behavior indirectly via a
    zero-bias module)."""
    torch.manual_seed(1)
    embed_dim, num_heads, L, N = 8, 2, 4, 2

    attn = nn.MultiheadAttention(embed_dim, num_heads, bias=False)
    attn.eval()
    x = torch.randn(L, N, embed_dim)

    with torch.no_grad():
        baseline_out, _ = manual_mha_forward(attn, x, ablate_heads=None)
        ablated_out, _ = manual_mha_forward(attn, x, ablate_heads=[0])
        all_ablated_out, _ = manual_mha_forward(attn, x, ablate_heads=list(range(num_heads)))

    assert ablated_out.shape == baseline_out.shape
    assert not torch.allclose(ablated_out, baseline_out), "ablating a head should change the output"
    # No bias anywhere (attn has bias=False and out_proj inherits that), so
    # zeroing every head's contribution before out_proj must zero the output.
    assert torch.allclose(all_ablated_out, torch.zeros_like(all_ablated_out), atol=1e-6)


def test_num_layers_and_heads_derives_from_module():
    """Sanity check the helper doesn't hardcode ViT-B/32's 12/12 anywhere —
    build a tiny fake transformer with a different shape and confirm it
    reports back exactly that shape."""
    class FakeBlock:
        def __init__(self, heads):
            self.attn = nn.MultiheadAttention(embed_dim=4, num_heads=heads)

    class FakeTransformer:
        def __init__(self, n_layers, n_heads):
            self.resblocks = [FakeBlock(n_heads) for _ in range(n_layers)]

    class FakeVisual:
        def __init__(self, n_layers, n_heads):
            self.transformer = FakeTransformer(n_layers, n_heads)

    fake_visual = FakeVisual(n_layers=3, n_heads=2)
    layers, heads = num_layers_and_heads(fake_visual)
    assert (layers, heads) == (3, 2)
