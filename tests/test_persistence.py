from features.persistence import (
    compute_feature_persistence,
    compute_persistence,
)


def test_full_persistence():
    deltas = {
        "F13": [0.2, 0.1, 0.05, 0.1],
    }

    result = compute_feature_persistence(deltas)

    assert result.score == 1.0
    assert result.get("F13") == 1.0


def test_partial_persistence():
    deltas = {
        "F13": [0.2, -0.1, 0.05, 0.1],
    }

    result = compute_feature_persistence(deltas)

    assert abs(result.score - 0.75) < 1e-9


def test_no_persistence():
    deltas = {
        "F13": [0.2, 0.1, -0.05],
    }

    result = compute_feature_persistence(deltas)

    assert abs(result.score - 1 / 3) < 1e-9


def test_multiple_features():
    deltas = {
        "F13": [0.2, 0.1, 0.1],
        "F14": [0.1, -0.1, 0.2],
    }

    result = compute_feature_persistence(deltas)

    assert result.get("F13") == 1.0
    assert abs(result.get("F14") - 2 / 3) < 1e-9
    assert abs(result.score - 5 / 6) < 1e-9


def test_zero_change_is_ignored():
    deltas = {
        "F13": [0.2, 0.0, 0.1],
    }

    result = compute_feature_persistence(deltas)

    assert result.get("F13") == 1.0


def test_final_zero():
    deltas = {
        "F13": [0.2, 0.1, 0.0],
    }

    result = compute_feature_persistence(deltas)

    assert result.get("F13") == 0.0


def test_empty_input():
    result = compute_feature_persistence({})

    assert result.score == 0.0
    assert result.feature_scores == {}


def test_alias_function():
    deltas = {
        "F13": [0.2, 0.1],
    }

    result1 = compute_feature_persistence(deltas)
    result2 = compute_persistence(deltas)

    assert result1 == result2


def test_score_range():
    deltas = {
        "F13": [0.2, -0.1, 0.3],
        "F14": [-0.2, -0.1, -0.3],
        "F16": [0.1, 0.2, 0.3],
    }

    result = compute_feature_persistence(deltas)

    assert 0.0 <= result.score <= 1.0