import math

import pytest

from features.move_transition import (
    F40MoveTransition,
    compute_move_transition,
    compute_move_transition_degree,
)


def test_single_feature_change():
    """1つの特徴量だけが変化した場合。"""

    current = object()
    future = object()

    feature_deltas = {
        "F13": 3.0,
    }

    result = compute_move_transition_degree(
        current,
        future,
        feature_deltas,
    )

    assert result == pytest.approx(3.0)


def test_two_feature_changes():
    """2つの特徴量の変化をユークリッド距離として統合する。"""

    current = object()
    future = object()

    feature_deltas = {
        "F13": 3.0,
        "F14": 4.0,
    }

    result = compute_move_transition_degree(
        current,
        future,
        feature_deltas,
    )

    assert result == pytest.approx(5.0)


def test_negative_changes_are_squared():
    """負の変化も二乗されるため、符号に依存しない。"""

    current = object()
    future = object()

    feature_deltas = {
        "F13": -3.0,
        "F14": 4.0,
    }

    result = compute_move_transition_degree(
        current,
        future,
        feature_deltas,
    )

    assert result == pytest.approx(5.0)


def test_zero_changes():
    """全特徴量の変化が0なら転換度も0になる。"""

    current = object()
    future = object()

    feature_deltas = {
        "F13": 0.0,
        "F14": 0.0,
        "F15": 0.0,
    }

    result = compute_move_transition_degree(
        current,
        future,
        feature_deltas,
    )

    assert result == pytest.approx(0.0)


def test_empty_feature_deltas():
    """特徴量変化が空の場合は0になる。"""

    current = object()
    future = object()

    result = compute_move_transition_degree(
        current,
        future,
        {},
    )

    assert result == pytest.approx(0.0)


def test_custom_feature_weights():
    """特徴量ごとのlambda_iを指定できる。"""

    current = object()
    future = object()

    feature_deltas = {
        "F13": 2.0,
        "F14": 1.0,
    }

    feature_weights = {
        "F13": 4.0,
        "F14": 1.0,
    }

    # sqrt(4 * 2^2 + 1 * 1^2)
    # = sqrt(17)
    result = compute_move_transition_degree(
        current,
        future,
        feature_deltas,
        feature_weights,
    )

    assert result == pytest.approx(math.sqrt(17.0))


def test_missing_weight_defaults_to_one():
    """重みが指定されていない特徴量は1.0として扱う。"""

    current = object()
    future = object()

    feature_deltas = {
        "F13": 3.0,
        "F14": 4.0,
    }

    feature_weights = {
        "F13": 1.0,
    }

    result = compute_move_transition_degree(
        current,
        future,
        feature_deltas,
        feature_weights,
    )

    assert result == pytest.approx(5.0)


def test_negative_weight_is_invalid():
    """特徴量の重みは負にできない。"""

    current = object()
    future = object()

    feature_deltas = {
        "F13": 1.0,
    }

    feature_weights = {
        "F13": -1.0,
    }

    with pytest.raises(ValueError):
        compute_move_transition_degree(
            current,
            future,
            feature_deltas,
            feature_weights,
        )


def test_current_position_is_required():
    """currentがNoneの場合はエラーにする。"""

    future = object()

    with pytest.raises(ValueError):
        compute_move_transition_degree(
            None,
            future,
            {"F13": 1.0},
        )


def test_future_position_is_required():
    """futureがNoneの場合はエラーにする。"""

    current = object()

    with pytest.raises(ValueError):
        compute_move_transition_degree(
            current,
            None,
            {"F13": 1.0},
        )


def test_detailed_result():
    """詳細結果には転換度と特徴量変化の両方が保持される。"""

    current = object()
    future = object()

    feature_deltas = {
        "F13": 3.0,
        "F14": 4.0,
    }

    result = compute_move_transition(
        current,
        future,
        feature_deltas,
    )

    assert isinstance(result, F40MoveTransition)
    assert result.degree == pytest.approx(5.0)
    assert result.get("F13") == pytest.approx(3.0)
    assert result.get("F14") == pytest.approx(4.0)


def test_degree_is_non_negative():
    """局面転換度は必ず0以上になる。"""

    current = object()
    future = object()

    feature_deltas = {
        "F13": -0.5,
        "F14": -0.8,
        "F15": 0.2,
    }

    result = compute_move_transition_degree(
        current,
        future,
        feature_deltas,
    )

    assert result >= 0.0