from pathlib import Path

import pytest

from interpreter.feature_weight_inference import FeatureWeightInference
from training.trained_model import LoadedFeatureWeightModel


MODEL_PATH = Path(
    "data/training_logs/"
    "linear_feature_weight_wcsc36_10games_logged/"
    "final_model.json"
)


def create_model() -> LoadedFeatureWeightModel:
    return LoadedFeatureWeightModel.from_json(MODEL_PATH)


def test_select_model_features_uses_model_order() -> None:
    model = create_model()
    inference = FeatureWeightInference(model)

    features = {
        "F33": 3.0,
        "F01": 1.0,
    }

    selected = inference.select_model_features(features)

    assert tuple(selected.keys()) == model.feature_names
    assert selected["F01"] == pytest.approx(1.0)
    assert selected["F33"] == pytest.approx(3.0)
    assert selected["F02"] == pytest.approx(0.0)


def test_predict_zero_features_equals_bias() -> None:
    model = create_model()
    inference = FeatureWeightInference(model)

    features = {}
    prediction = inference.predict(features)

    assert prediction == pytest.approx(model.bias)


def test_predict_matches_direct_model_prediction() -> None:
    model = create_model()
    inference = FeatureWeightInference(model)

    features = {
        name: 0.0
        for name in model.feature_names
    }
    features["F33"] = 1.0

    selected = inference.select_model_features(features)
    expected = model.predict(selected)
    actual = inference.predict(features)

    assert actual == pytest.approx(expected)


def test_strict_mode_rejects_missing_features() -> None:
    model = create_model()
    inference = FeatureWeightInference(model)

    with pytest.raises(KeyError, match="Missing model features"):
        inference.select_model_features(
            {"F01": 1.0},
            strict=True,
        )


def test_predict_with_details_has_contributions() -> None:
    model = create_model()
    inference = FeatureWeightInference(model)

    features = {
        name: 0.0
        for name in model.feature_names
    }
    features["F33"] = 1.0

    result = inference.predict_with_details(features)

    assert result["prediction"] == pytest.approx(
        model.predict(features)
    )
    assert result["bias"] == pytest.approx(model.bias)
    assert result["features"]["F33"] == pytest.approx(1.0)
    assert result["contributions"]["F33"] == pytest.approx(
        model.weights[model.feature_names.index("F33")]
    )
