from features.strategy import compute_strategy_direction


def test_strategy_direction_basic():
    deltas = {
        "F13": 0.4,
        "F14": 0.2,
        "F15": 0.4,
        "F16": 0.3,
        "F28": 0.1,
    }

    result = compute_strategy_direction(deltas)

    assert abs(result.material - 0.4) < 1e-9
    assert abs(result.attack - 0.3) < 1e-9
    assert abs(result.activity - 0.3) < 1e-9
    assert abs(result.formation - 0.1) < 1e-9
    assert abs(result.defense - 0.0) < 1e-9


def test_strategy_direction_multiple_features():
    deltas = {
        "F14": 0.2,
        "F15": 0.4,
        "F19": 0.6,
    }

    result = compute_strategy_direction(deltas)

    assert abs(result.attack - 0.4) < 1e-9


def test_strategy_direction_missing_features():
    result = compute_strategy_direction({})

    assert result.vector == (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def test_strategy_direction_negative():
    deltas = {
        "F13": -0.5,
        "F14": -0.2,
        "F16": -0.3,
    }

    result = compute_strategy_direction(deltas)

    assert result.material == -0.5
    assert result.attack == -0.2
    assert result.activity == -0.3


def test_strategy_direction_range():
    deltas = {
        "F13": 1.0,
        "F14": 1.0,
        "F15": 1.0,
        "F16": 1.0,
        "F28": 1.0,
    }

    result = compute_strategy_direction(deltas)

    for value in result.vector:
        assert -1.0 <= value <= 1.0