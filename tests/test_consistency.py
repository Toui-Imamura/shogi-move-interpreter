from features.consistency import (
    compute_consistency,
    compute_feature_consistency,
)


def test_all_same_direction():
    deltas = {
        "F13": 0.4,
        "F14": 0.2,
        "F15": 0.3,
        "F16": 0.5,
        "F28": 0.1,
    }

    result = compute_consistency(deltas)

    assert result.score == 1.0


def test_opposite_direction():
    deltas = {
        "F13": 0.4,
        "F14": -0.2,
    }

    result = compute_consistency(deltas)

    assert result.score == 0.0


def test_partial_consistency():
    deltas = {
        "F13": 0.4,
        "F14": 0.2,
        "F16": -0.3,
    }

    result = compute_consistency(deltas)

    assert abs(result.score - 1 / 3) < 1e-9


def test_zero_values_are_ignored():
    deltas = {
        "F13": 0.4,
        "F14": 0.0,
        "F15": 0.0,
    }

    result = compute_consistency(deltas)

    assert result.score == 1.0


def test_no_features():
    result = compute_consistency({})

    assert result.score == 1.0


def test_result_vector():
    deltas = {
        "F13": 0.4,
        "F14": 0.2,
        "F16": -0.3,
        "F28": 0.1,
    }

    result = compute_consistency(deltas)

    assert len(result.vector) == 5
    assert result.material == 0.4
    assert abs(result.attack - 0.2) < 1e-9
    assert result.defense == 0.0
    assert result.activity == -0.3
    assert abs(result.formation - 0.1) < 1e-9


def test_score_range():
    deltas = {
        "F13": -0.5,
        "F14": 0.2,
        "F16": 0.3,
        "F28": -0.1,
    }

    result = compute_consistency(deltas)

    assert 0.0 <= result.score <= 1.0


def test_alias_function():
    deltas = {
        "F13": 0.4,
        "F14": 0.2,
    }

    result1 = compute_consistency(deltas)
    result2 = compute_feature_consistency(deltas)

    assert result1 == result2