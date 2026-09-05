import pytest

from features.change_concentration import (
    compute_feature_change_concentration,
)


def test_single_feature_change():
    """1つの特徴量だけが変化した場合、集中度は1.0になる。"""

    feature_deltas = {
        "F13": 1.0,
        "F14": 0.0,
        "F15": 0.0,
    }

    result = compute_feature_change_concentration(feature_deltas)

    assert result == pytest.approx(1.0)


def test_two_equal_changes():
    """2つの特徴量が同じ大きさで変化した場合、集中度は0.5になる。"""

    feature_deltas = {
        "F13": 1.0,
        "F14": 1.0,
    }

    result = compute_feature_change_concentration(feature_deltas)

    assert result == pytest.approx(0.5)


def test_four_equal_changes():
    """4つの特徴量が同じ大きさで変化した場合、集中度は0.25になる。"""

    feature_deltas = {
        "F13": 1.0,
        "F14": 1.0,
        "F15": 1.0,
        "F16": 1.0,
    }

    result = compute_feature_change_concentration(feature_deltas)

    assert result == pytest.approx(0.25)


def test_negative_changes_use_absolute_value():
    """負の変化も絶対値で扱う。"""

    feature_deltas = {
        "F13": -1.0,
        "F14": 1.0,
    }

    result = compute_feature_change_concentration(feature_deltas)

    assert result == pytest.approx(0.5)


def test_different_change_sizes():
    """変化量の大きさに応じて集中度が変化する。"""

    feature_deltas = {
        "F13": 2.0,
        "F14": 1.0,
    }

    # p13 = 2/3
    # p14 = 1/3
    # concentration = 4/9 + 1/9 = 5/9
    result = compute_feature_change_concentration(feature_deltas)

    assert result == pytest.approx(5.0 / 9.0)


def test_zero_changes():
    """全特徴量が変化していない場合は0.0になる。"""

    feature_deltas = {
        "F13": 0.0,
        "F14": 0.0,
        "F15": 0.0,
    }

    result = compute_feature_change_concentration(feature_deltas)

    assert result == pytest.approx(0.0)


def test_empty_input():
    """空の入力では0.0になる。"""

    result = compute_feature_change_concentration({})

    assert result == pytest.approx(0.0)


def test_result_is_between_zero_and_one():
    """集中度は必ず0～1の範囲に収まる。"""

    feature_deltas = {
        "F13": 0.8,
        "F14": -0.3,
        "F15": 0.2,
        "F16": -0.1,
    }

    result = compute_feature_change_concentration(feature_deltas)

    assert 0.0 <= result <= 1.0


def test_zero_values_do_not_affect_concentration():
    """変化量0の特徴量を追加しても集中度は変わらない。"""

    feature_deltas_without_zero = {
        "F13": 1.0,
        "F14": 1.0,
    }

    feature_deltas_with_zero = {
        "F13": 1.0,
        "F14": 1.0,
        "F15": 0.0,
        "F16": 0.0,
    }

    result_without_zero = compute_feature_change_concentration(
        feature_deltas_without_zero
    )

    result_with_zero = compute_feature_change_concentration(
        feature_deltas_with_zero
    )

    assert result_without_zero == pytest.approx(0.5)
    assert result_with_zero == pytest.approx(0.5)