"""Bundled inference must work offline, independent of working directory."""
import numpy as np

from recognizer.learned_ink import FEATURE_NAMES, load_model, predict_features


def test_ink_model_loads_and_predicts_from_an_unrelated_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    load_model.cache_clear()
    model = load_model()
    assert model['features'] == FEATURE_NAMES
    features = np.zeros((3, 4, len(FEATURE_NAMES)), np.float32)
    probabilities = predict_features(features, model)
    assert probabilities.shape == (3, 4)
    assert np.isfinite(probabilities).all()
    assert np.all((0 <= probabilities) & (probabilities <= 1))
