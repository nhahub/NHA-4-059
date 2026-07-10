import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from PIL import Image

from src.xai.attention_rollout import AttentionRollout, cls_relevance, rollout


def test_rollout_matches_hand_computed_example():
    """The single highest-value test in this feature: verifies rollout()'s
    matrix-multiplication order is correct, not just its output shape.

    2 layers, 3 tokens (1 CLS + 2 patches):
      Layer 0: CLS attends 50/50 to itself and patch1, ignores patch2.
               Patches trivially self-attend (irrelevant to the CLS
               readout, but every attention matrix must be row-stochastic).
      Layer 1: CLS attends 20% self, 30% patch1, 50% patch2.

    Fusing each with the residual (0.5*A + 0.5*I) and composing them in
    the correct order (layer 0 applied first / rightmost) gives, by hand:

        fused0 = [[0.75, 0.25, 0   ],   fused1 = [[0.6, 0.15, 0.25],
                  [0,    1,    0   ],              [0,   1,    0   ],
                  [0,    0,    1   ]]               [0,   0,    1   ]]

        rolled = fused1 @ fused0
               = [[0.45, 0.30, 0.25],
                  [0,    1,    0   ],
                  [0,    0,    1   ]]

    So CLS's relevance over [patch1, patch2] should be exactly [0.30, 0.25].
    Getting the composition order backwards (fused0 @ fused1 instead) gives
    a different but still row-stochastic, still-plausible-looking
    [0.3625, 0.1875] instead -- silently wrong, no error, which is exactly
    why this needs a numeric assertion rather than a shape/range check.
    """
    a0 = torch.tensor(
        [
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    a1 = torch.tensor(
        [
            [0.2, 0.3, 0.5],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    rolled = rollout([a0, a1])
    assert torch.allclose(rolled.sum(dim=-1), torch.ones(3), atol=1e-6), "rows must stay row-stochastic"

    expected_row0 = torch.tensor([0.45, 0.30, 0.25])
    assert torch.allclose(rolled[0], expected_row0, atol=1e-6)

    rel = cls_relevance(rolled, num_patches=2)
    assert torch.allclose(rel, torch.tensor([0.30, 0.25]), atol=1e-6)


def test_rollout_wrong_order_gives_different_result():
    """Companion check: confirms the 'get it backwards' failure mode
    described above is real and would actually be caught -- i.e. this test
    file isn't accidentally insensitive to the composition order."""
    a0 = torch.tensor([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    a1 = torch.tensor([[0.2, 0.3, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    correct = rollout([a0, a1])
    backwards = rollout([a1, a0])  # layers passed in the wrong order
    assert not torch.allclose(correct, backwards)


class _FakeBlock(nn.Module):
    """Minimal stand-in for CLIP's ResidualAttentionBlock: just the
    self-attention call with the same need_weights=False/attn_mask=None
    signature, skipping MLP/LayerNorm/residual entirely since this test
    only exercises the attention-capture + rollout wiring, not CLIP
    fidelity (that needs a real CLIP ViT, exercised in Colab per the
    project plan)."""

    def __init__(self, embed_dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, heads)

    def forward(self, x):
        return self.attn(x, x, x, need_weights=False, attn_mask=None)[0]


class _FakeTransformer(nn.Module):
    def __init__(self, resblocks):
        super().__init__()
        self.resblocks = nn.ModuleList(resblocks)

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        return x


class _FakeVisual(nn.Module):
    """Fake ViT visual backbone: real nn.MultiheadAttention blocks (so the
    monkeypatch-based capture in vit_attention_utils.py runs against real
    PyTorch attention internals), but synthetic patch tokens instead of a
    real patch-embedding conv, since this test is about validating the
    capture+rollout wiring end-to-end, not reproducing CLIP itself."""

    def __init__(self, num_layers, embed_dim, heads, num_patches):
        super().__init__()
        self.conv1 = nn.Conv2d(3, embed_dim, kernel_size=1)  # only .weight.dtype is used
        self.transformer = _FakeTransformer([_FakeBlock(embed_dim, heads) for _ in range(num_layers)])
        self.num_tokens = num_patches + 1  # +1 for CLS
        self.embed_dim = embed_dim

    def forward(self, x):
        n = x.shape[0]
        torch.manual_seed(0)
        tokens = torch.randn(self.num_tokens, n, self.embed_dim, dtype=x.dtype, device=x.device)
        return self.transformer(tokens)


class _FakeCLIPModel:
    def __init__(self, visual):
        self.model = type("M", (), {"visual": visual})()
        self.preprocess = lambda img: torch.zeros(3, 4, 4)
        self.device = "cpu"


def test_attention_rollout_end_to_end_with_fake_transformer():
    """Integration test for AttentionRollout.__call__: real attention
    modules, fake CLIP plumbing around them. Confirms the monkeypatch
    capture in vit_attention_utils.py actually collects one weight matrix
    per layer and that AttentionRollout wires it into a valid (224,224)
    map -- without needing real CLIP-B/32 weights."""
    visual = _FakeVisual(num_layers=4, embed_dim=8, heads=2, num_patches=9)  # 3x3 grid
    fake_clip_model = _FakeCLIPModel(visual)

    rollout_explainer = AttentionRollout(fake_clip_model)
    cam = rollout_explainer(Image.new("RGB", (32, 32)))

    assert cam.shape == (224, 224)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0 + 1e-6


def test_attention_rollout_rejects_conv_backbone():
    class _ConvVisual:
        pass  # no .transformer attribute

    fake_clip_model = _FakeCLIPModel(_ConvVisual())
    with pytest.raises(TypeError):
        AttentionRollout(fake_clip_model)
