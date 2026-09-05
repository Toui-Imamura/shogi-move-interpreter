from features.mcts_weighted import (
    MCTSVariation,
    compute_visit_weighted_change,
)


def test_single_feature_weighted_change():
    variations = [
        MCTSVariation(
            visits=10,
            deltas={"F13": 0.2},
        ),
        MCTSVariation(
            visits=5,
            deltas={"F13": 0.1},
        ),
    ]

    result = compute_visit_weighted_change(
        variations
    )

    expected = (10 * 0.2 + 5 * 0.1) / 15

    assert abs(
        result.get("F13") - expected
    ) < 1e-9


def test_multiple_features():
    variations = [
        MCTSVariation(
            visits=10,
            deltas={
                "F13": 0.2,
                "F14": 0.1,
            },
        ),
        MCTSVariation(
            visits=5,
            deltas={
                "F13": 0.1,
                "F14": 0.0,
            },
        ),
        MCTSVariation(
            visits=20,
            deltas={
                "F13": 0.0,
                "F14": -0.2,
            },
        ),
    ]

    result = compute_visit_weighted_change(
        variations
    )

    expected_f13 = (
        10 * 0.2
        + 5 * 0.1
        + 20 * 0.0
    ) / 35

    expected_f14 = (
        10 * 0.1
        + 5 * 0.0
        + 20 * (-0.2)
    ) / 35

    assert abs(
        result.get("F13") - expected_f13
    ) < 1e-9

    assert abs(
        result.get("F14") - expected_f14
    ) < 1e-9


def test_high_visit_variation_has_more_influence():
    variations = [
        MCTSVariation(
            visits=1,
            deltas={"F13": 1.0},
        ),
        MCTSVariation(
            visits=9,
            deltas={"F13": 0.0},
        ),
    ]

    result = compute_visit_weighted_change(
        variations
    )

    assert abs(
        result.get("F13") - 0.1
    ) < 1e-9


def test_equal_visits_are_simple_average():
    variations = [
        MCTSVariation(
            visits=10,
            deltas={"F13": 0.2},
        ),
        MCTSVariation(
            visits=10,
            deltas={"F13": 0.4},
        ),
    ]

    result = compute_visit_weighted_change(
        variations
    )

    assert abs(
        result.get("F13") - 0.3
    ) < 1e-9


def test_missing_feature_is_zero():
    variations = [
        MCTSVariation(
            visits=10,
            deltas={"F13": 0.2},
        ),
        MCTSVariation(
            visits=10,
            deltas={},
        ),
    ]

    result = compute_visit_weighted_change(
        variations
    )

    assert abs(
        result.get("F13") - 0.1
    ) < 1e-9


def test_negative_change():
    variations = [
        MCTSVariation(
            visits=10,
            deltas={"F13": -0.2},
        ),
        MCTSVariation(
            visits=10,
            deltas={"F13": -0.4},
        ),
    ]

    result = compute_visit_weighted_change(
        variations
    )

    assert abs(
        result.get("F13") - (-0.3)
    ) < 1e-9


def test_empty_variations():
    result = compute_visit_weighted_change([])

    assert result.weighted_changes == {}


def test_zero_total_visits():
    variations = [
        MCTSVariation(
            visits=0,
            deltas={"F13": 0.5},
        ),
        MCTSVariation(
            visits=0,
            deltas={"F13": 0.2},
        ),
    ]

    result = compute_visit_weighted_change(
        variations
    )

    assert result.weighted_changes == {}


def test_result_is_float():
    variations = [
        MCTSVariation(
            visits=10,
            deltas={"F13": 0.2},
        ),
    ]

    result = compute_visit_weighted_change(
        variations
    )

    assert isinstance(
        result.get("F13"),
        float,
    )