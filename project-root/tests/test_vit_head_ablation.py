import numpy as np
import pytest
from PIL import Image, ImageDraw

torch = pytest.importorskip("torch")
nn = torch.nn

from src.models.vit_head_ablation import ViTHeadAblationMitigator


class _FakeBlock(nn.Module):
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
    """Unlike test_attention_rollout.py's fake (which uses fixed-seed random
    tokens, fine for shape/wiring checks), this one derives tokens from the
    input's actual per-channel mean color through a fixed random linear
    projection -- so a "logo" that visibly changes the image (e.g. a pasted
    colored rectangle) produces genuinely different tokens/attention,
    letting compute_head_sensitivity's ranking mean something rather than
    always seeing identical clean/logo inputs."""

    def __init__(self, num_layers, embed_dim, heads, num_tokens):
        super().__init__()
        self.conv1 = nn.Conv2d(3, embed_dim, kernel_size=1)
        self.transformer = _FakeTransformer([_FakeBlock(embed_dim, heads) for _ in range(num_layers)])
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        torch.manual_seed(42)
        self.proj = nn.Linear(3, embed_dim * num_tokens)

    def forward(self, x):
        n = x.shape[0]
        channel_mean = x.mean(dim=(2, 3))  # (N, 3) -- depends on actual pixel content
        tokens_flat = self.proj(channel_mean)  # (N, embed_dim * num_tokens)
        tokens = tokens_flat.reshape(n, self.num_tokens, self.embed_dim).permute(1, 0, 2)
        return self.transformer(tokens)


class _FakeCLIPModel:
    def __init__(self, visual):
        self.model = type("M", (), {"visual": visual})()
        self.device = "cpu"

    @staticmethod
    def preprocess(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.resize((8, 8)), dtype=np.float32) / 255.0
        return torch.tensor(arr).permute(2, 0, 1)


def _paste_fake_logo(img: Image.Image) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, out.width, out.height // 3], fill=(0, 200, 0))
    return out


def _build_mitigator(num_layers=3, heads=2, num_patches=8):
    visual = _FakeVisual(num_layers=num_layers, embed_dim=8, heads=heads, num_tokens=num_patches + 1)
    return ViTHeadAblationMitigator(_FakeCLIPModel(visual)), visual


def test_rejects_conv_backbone():
    class _ConvVisual:
        pass

    fake_clip_model = _FakeCLIPModel.__new__(_FakeCLIPModel)
    fake_clip_model.model = type("M", (), {"visual": _ConvVisual()})()
    fake_clip_model.device = "cpu"
    with pytest.raises(TypeError):
        ViTHeadAblationMitigator(fake_clip_model)


def test_compute_head_sensitivity_shape_and_completeness():
    mitigator, _ = _build_mitigator(num_layers=3, heads=2)
    examples = [{"image": Image.new("RGB", (32, 32), (r, 100, 150))} for r in (10, 60, 120, 200)]

    ranked = mitigator.compute_head_sensitivity(
        examples,
        decode_image=lambda img: img,
        apply_logo=_paste_fake_logo,
        sample_size=4,
    )

    assert len(ranked) == 3 * 2  # num_layers * num_heads
    assert set(ranked) == {(l, h) for l in range(3) for h in range(2)}, "must be a permutation of all pairs"


def test_apply_mitigation_changes_output_and_remove_restores_it():
    mitigator, visual = _build_mitigator(num_layers=3, heads=2)
    examples = [{"image": Image.new("RGB", (32, 32), (r, 100, 150))} for r in (10, 60, 120, 200)]
    mitigator.compute_head_sensitivity(
        examples,
        decode_image=lambda img: img,
        apply_logo=_paste_fake_logo,
        sample_size=4,
    )

    probe_img = Image.new("RGB", (32, 32), (77, 88, 99))
    img_t = mitigator.clip_model.preprocess(probe_img).unsqueeze(0)

    with torch.no_grad():
        original_out = visual(img_t)

    mitigator.apply_mitigation(num_heads_to_ablate=2)
    with torch.no_grad():
        ablated_out = visual(img_t)
    assert not torch.allclose(original_out, ablated_out), "ablation should change the output"

    mitigator.remove_mitigation()
    with torch.no_grad():
        restored_out = visual(img_t)
    assert torch.allclose(original_out, restored_out, atol=1e-5), "remove_mitigation should fully restore"
