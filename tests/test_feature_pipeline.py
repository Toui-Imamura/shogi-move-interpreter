import pytest

from interpreter.feature_pipeline import (
    FeaturePipelineResult,
    FeatureValuePair,
    compute_feature_deltas,
)


def test_feature_value_pair_delta():
    """未来値 - 現在値として変化量が計算される。"""

    value = FeatureValuePair(
        current=0.2,
        future=0.7,
    )

    assert value.delta == pytest.approx(0.5)


def test_negative_delta():
    """特徴量が減少した場合は負の変化量になる。"""

    value = FeatureValuePair(
        current=0.8,
        future=0.3,
    )

    assert value.delta == pytest.approx(-0.5)


def test_compute_single_feature():
    """1つの特徴量を計算できる。"""

    current = {"value": 1.0}
    future = {"value": 3.0}

    def extractor(position):
        return position["value"]

    result = compute_feature_deltas(
        current,
        future,
        {"F13": extractor},
    )

    assert isinstance(result, FeaturePipelineResult)
    assert result.get("F13") is not None
    assert result.get("F13").current == pytest.approx(1.0)
    assert result.get("F13").future == pytest.approx(3.0)
    assert result.get("F13").delta == pytest.approx(2.0)


def test_compute_multiple_features():
    """複数特徴量をまとめて計算できる。"""

    current = {
        "attack": 0.2,
        "defense": 0.5,
    }

    future = {
        "attack": 0.7,
        "defense": 0.3,
    }

    def attack_extractor(position):
        return position["attack"]

    def defense_extractor(position):
        return position["defense"]

    extractors = {
        "F19": attack_extractor,
        "F23": defense_extractor,
    }

    result = compute_feature_deltas(
        current,
        future,
        extractors,
    )

    assert result.deltas["F19"] == pytest.approx(0.5)
    assert result.deltas["F23"] == pytest.approx(-0.2)


def test_result_deltas():
    """deltasプロパティから変化量をまとめて取得できる。"""

    current = {
        "a": 1.0,
        "b": 2.0,
    }

    future = {
        "a": 1.5,
        "b": 1.0,
    }

    extractors = {
        "F13": lambda position: position["a"],
        "F14": lambda position: position["b"],
    }

    result = compute_feature_deltas(
        current,
        future,
        extractors,
    )

    assert result.deltas == {
        "F13": pytest.approx(0.5),
        "F14": pytest.approx(-1.0),
    }


def test_empty_extractors():
    """特徴量計算器が空なら空の結果になる。"""

    current = object()
    future = object()

    result = compute_feature_deltas(
        current,
        future,
        {},
    )

    assert result.deltas == {}


def test_current_position_required():
    """currentがNoneの場合はエラーになる。"""

    future = object()

    with pytest.raises(ValueError):
        compute_feature_deltas(
            None,
            future,
            {},
        )


def test_future_position_required():
    """futureがNoneの場合はエラーになる。"""

    current = object()

    with pytest.raises(ValueError):
        compute_feature_deltas(
            current,
            None,
            {},
        )


def test_integer_values_are_converted_to_float():
    """特徴量の戻り値はfloatとして保持する。"""

    current = {"value": 1}
    future = {"value": 4}

    def extractor(position):
        return position["value"]

    result = compute_feature_deltas(
        current,
        future,
        {"F13": extractor},
    )

    feature = result.get("F13")

    assert isinstance(feature.current, float)
    assert isinstance(feature.future, float)
    assert feature.delta == pytest.approx(3.0)