from features.mcts_frequency import (
    compute_mcts_change_frequency,
)


def test_single_feature_frequency():
    variations = [
        {"F13": 0.2},
        {"F13": 0.1},
        {"F13": 0.0},
        {"F13": -0.3},
    ]

    result = compute_mcts_change_frequency(variations)

    assert abs(result.get("F13") - 0.75) < 1e-9


def test_multiple_feature_frequency():
    variations = [
        {"F13": 0.2, "F14": 0.1},
        {"F13": 0.1, "F14": 0.0},
        {"F13": 0.0, "F14": -0.2},
        {"F13": 0.3, "F14": 0.5},
    ]

    result = compute_mcts_change_frequency(variations)

    assert abs(result.get("F13") - 0.75) < 1e-9
    assert abs(result.get("F14") - 0.75) < 1e-9


def test_zero_change_is_not_counted():
    variations = [
        {"F13": 0.0},
        {"F13": 0.0},
        {"F13": 0.2},
        {"F13": -0.1},
    ]

    result = compute_mcts_change_frequency(variations)

    assert abs(result.get("F13") - 0.50) < 1e-9


def test_missing_feature_is_treated_as_no_change():
    variations = [
        {"F13": 0.2},
        {"F14": 0.1},
        {"F13": 0.3},
        {},
    ]

    result = compute_mcts_change_frequency(variations)

    assert abs(result.get("F13") - 0.50) < 1e-9
    assert abs(result.get("F14") - 0.25) < 1e-9


def test_all_variations_changed():
    variations = [
        {"F13": 0.1},
        {"F13": -0.2},
        {"F13": 0.3},
        {"F13": 0.4},
    ]

    result = compute_mcts_change_frequency(variations)

    assert abs(result.get("F13") - 1.0) < 1e-9


def test_no_variations():
    result = compute_mcts_change_frequency([])

    assert result.feature_frequencies == {}


def test_frequency_range():
    variations = [
        {"F13": 0.1},
        {"F13": 0.0},
        {"F13": -0.2},
    ]

    result = compute_mcts_change_frequency(variations)

    assert 0.0 <= result.get("F13") <= 1.0


def test_epsilon():
    variations = [
        {"F13": 1e-10},
        {"F13": 1e-7},
    ]

    result = compute_mcts_change_frequency(
        variations,
        epsilon=1e-8,
    )

    assert abs(result.get("F13") - 0.50) < 1e-9


def test_negative_epsilon():
    variations = [
        {"F13": 0.1},
    ]

    try:
        compute_mcts_change_frequency(
            variations,
            epsilon=-1e-8,
        )
        assert False
    except ValueError:
        pass