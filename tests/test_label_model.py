import numpy as np

from recognizer.label_model import FEATURE_NAMES, load_model, predict_features


def test_tree_export_keeps_double_precision_threshold_comparison():
    # Rounding the threshold to float32 incorrectly sends x=1 to the left.
    threshold = float(np.nextafter(np.float64(1),np.float64(0)))
    model = {'trees': [[[0,threshold,1,2,.5],[-1,0,-1,-1,.1],[-1,0,-1,-1,.9]]]}
    probabilities = predict_features(np.asarray([[1.],[.5]],np.float32),model)
    assert np.allclose(probabilities,[.9,.1])


def test_packaged_model_loads_independently_of_current_directory(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    load_model.cache_clear()
    model = load_model()
    assert model is not None
    assert model['feature_names'] == list(FEATURE_NAMES)
    probabilities = predict_features(np.zeros((3,len(FEATURE_NAMES)),np.float32),model)
    assert probabilities.shape == (3,)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert len(predict_features(np.empty((0,len(FEATURE_NAMES)),np.float32),model)) == 0
