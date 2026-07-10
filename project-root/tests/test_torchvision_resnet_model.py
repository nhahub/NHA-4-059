import pickle

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.models.torchvision_resnet_model import SupervisedResNetModel


def _build_model(tmp_path, with_classifier=True):
    """weights=None (random init, no download) is enough to verify the
    adapter's shape/interface -- exactly the workaround used in
    docs/lrp_setup_notes.md when this sandbox's network couldn't reach
    download.pytorch.org. Real pretrained weights are a Colab-side concern."""
    classifier_path = None
    if with_classifier:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression()
        X = np.random.RandomState(0).randn(40, 2048)
        y = np.array([i % 8 for i in range(40)])
        clf.fit(X, y)
        classifier_path = tmp_path / "clf.pkl"
        with open(classifier_path, "wb") as f:
            pickle.dump(clf, f)

    return SupervisedResNetModel(classifier_path=classifier_path, device="cpu", weights=None)


def test_shape_matches_clip_model_adapter_contract(tmp_path):
    model = _build_model(tmp_path)

    assert model.device == "cpu"
    assert model.architecture == "resnet"
    assert hasattr(model.model, "visual")
    assert hasattr(model.model, "encode_image")
    assert hasattr(model.model.visual, "conv1"), "LRP's _ClassifierHead needs .visual.conv1"

    modules = dict(model.model.visual.named_modules())
    assert "relu" in modules, "CHMitigator needs a top-level early-layer name for this backbone"
    assert isinstance(modules["relu"], torch.nn.ReLU)

    last_conv_names = [n for n, m in model.model.visual.named_modules() if isinstance(m, torch.nn.Conv2d)]
    assert last_conv_names[-1] == "layer4.2.conv3", "GradCAM hooks the block containing this"


def test_encode_image_returns_2048d_vector(tmp_path):
    model = _build_model(tmp_path)
    img = Image.new("RGB", (300, 300), (10, 80, 200))

    feat = model.encode_image(img)
    assert feat.shape == (2048,)
    assert np.isfinite(feat).all()


def test_predict_uses_loaded_classifier(tmp_path):
    model = _build_model(tmp_path)
    img = Image.new("RGB", (300, 300), (10, 80, 200))
    feat = model.encode_image(img)

    pred = model.predict(feat)
    assert pred.shape == (1,)
    assert 0 <= pred[0] < 8


def test_predict_without_classifier_raises(tmp_path):
    model = _build_model(tmp_path, with_classifier=False)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros(2048))


def test_encode_image_tensor_is_grad_enabled(tmp_path):
    model = _build_model(tmp_path)
    img = Image.new("RGB", (300, 300), (10, 80, 200))

    feat, img_t = model.encode_image_tensor(img)
    assert img_t.requires_grad
    assert feat.shape[-1] == 2048
