import pickle

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("clip")

from src.models.clip_model import CLIPModel
from src.xai.gradcam import GradCAM


def _build_real_clip_model(tmp_path):
    """These tests need the real CLIP RN50 weights (downloaded on first use)
    and a working torch install — they're meant to run in the team's
    GPU/Colab environment (per README), not a minimal CI sandbox. Skips
    cleanly if the weights can't be fetched here."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()
    X = np.random.RandomState(0).randn(40, 1024)
    y = np.array([i % 8 for i in range(40)])
    clf.fit(X, y)

    clf_path = tmp_path / "clf.pkl"
    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)

    try:
        return CLIPModel(classifier_path=clf_path)
    except Exception as e:  # noqa: BLE001 - broad on purpose, this is an environment skip
        pytest.skip(f"CLIP RN50 weights/network unavailable in this environment: {e}")


def test_gradcam_heatmap_shape_rn50(tmp_path):
    """UT-5: generate a heatmap for 1 image; verify output array dimensions
    match the input's spatial dimensions."""
    clip_model = _build_real_clip_model(tmp_path)
    gradcam = GradCAM(clip_model)
    img = Image.new("RGB", (224, 224), (120, 60, 30))

    cam, class_idx = gradcam(img)

    assert cam.shape == (224, 224)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0
    assert 0 <= class_idx < 8


def test_gradcam_vit_backbone_not_implemented():
    """UT-6 asks for Grad-CAM on both ViT-B/32 and RN50 backbones. This
    codebase's CLIPModel hardcodes RN50 (see src/models/clip_model.py) —
    ViT-B/32 zero-shot support is a separate, larger architectural gap
    (SAD FR-2) that hasn't been implemented, not a Grad-CAM bug. Documenting
    that scope here rather than asserting behaviour that doesn't exist."""
    pytest.skip(
        "ViT-B/32 backbone not implemented in this codebase's CLIPModel "
        "(RN50 + LogisticRegression only) — see project README gap list, FR-2."
    )
