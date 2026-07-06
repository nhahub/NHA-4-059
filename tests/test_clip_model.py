import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression

from src.models.clip_model import CLIPModel


def _fake_clip_model_with_classifier():
    """Builds a CLIPModel instance without going through __init__ (which
    loads real CLIP weights via torch) — predict() only ever touches
    self.classifier, so this is enough to unit-test it in isolation."""
    clf = LogisticRegression()
    X = np.random.RandomState(0).randn(20, 8)
    y = np.array([i % 4 for i in range(20)])
    clf.fit(X, y)

    model = CLIPModel.__new__(CLIPModel)
    model.classifier = clf
    return model, X


def test_predict_returns_valid_class_indices():
    """UT-3: run predict() on preprocessed features; verify output types."""
    model, X = _fake_clip_model_with_classifier()
    preds = model.predict(X[:5])

    assert len(preds) == 5
    assert set(preds).issubset({0, 1, 2, 3})


def test_predict_accepts_a_single_1d_feature_vector():
    model, X = _fake_clip_model_with_classifier()
    preds = model.predict(X[0])

    assert len(preds) == 1


def test_predict_without_classifier_raises_gracefully():
    """UT-4 (adapted): the SAD's UT-4 exercises predict(image, prompts) with
    an empty prompts list, which belongs to the zero-shot
    CLIPModel.predict(image, prompts) contract in SAD §17. This codebase's
    CLIPModel.predict(features) is features-based (a LogisticRegression
    probe on CLIP embeddings, not zero-shot text-prompt classification —
    see README's gap list), so the equivalent graceful-failure path here is
    calling predict() before a classifier has been loaded."""
    model = CLIPModel.__new__(CLIPModel)
    model.classifier = None

    with pytest.raises(RuntimeError):
        model.predict(np.zeros((1, 8)))
