import pytest

from features.normalization import (
    tanh_normalize,
    normalize_feature_deltas,
)


def test_tanh_normalize_zero():
    assert tanh_normalize(
        0.0,
        1.0,
    ) == pytest.approx(0.0)


def test_tanh_normalize_positive():
    value = tanh_normalize(
        1.0,
        1.0,
    )

    assert 0.0 < value < 1.0


def test_tanh_normalize_negative():
    value = tanh_normalize(
        -1.0,
        1.0,
    )

    assert -1.0 < value < 0.0


def test_tanh_normalize_range():

    for value in [
        -100.0,
        -10.0,
        -1.0,
        0.0,
        1.0,
        10.0,
        100.0,
    ]:
        normalized = tanh_normalize(
            value,
            1.0,
        )

        assert -1.0 <= normalized <= 1.0


def test_invalid_scale():

    with pytest.raises(ValueError):
        tanh_normalize(
            1.0,
            0.0,
        )


def test_normalize_feature_deltas():

    result = normalize_feature_deltas(
        {
            "F06": 3.0,
            "F11": 0.5,
        },
        {
            "F06": 3.0,
            "F11": 0.5,
        },
    )

    assert result["F06"] == pytest.approx(
        0.761594,
        abs=1e-5,
    )

    assert result["F11"] == pytest.approx(
        0.761594,
        abs=1e-5,
    )