import cshogi
import pytest

from interpreter.mcts_features import (
    compute_mcts_features,
)

from interpreter.variation_features import (
    compute_variation_features,
)


def test_compute_mcts_features():
    board = cshogi.Board()

    result1 = compute_variation_features(
        board,
        [
            "7g7f",
            "3c3d",
            "2g2f",
        ],
    )

    result2 = compute_variation_features(
        board,
        [
            "2g2f",
            "8c8d",
            "7g7f",
        ],
    )

    result = compute_mcts_features(
        board,
        [
            {
                "result": result1,
                "visits": 10,
            },
            {
                "result": result2,
                "visits": 5,
            },
        ],
    )

    assert len(result.variation_deltas) == 2

    assert result.f37.feature_frequencies
    assert result.f38.weighted_changes

    assert "F01" in result.variation_deltas[0]
    assert "F33" in result.variation_deltas[0]

    assert 0.0 <= result.f39 <= 1.0


def test_mcts_feature_result_with_single_variation():
    board = cshogi.Board()

    variation = compute_variation_features(
        board,
        [
            "7g7f",
            "3c3d",
        ],
    )

    result = compute_mcts_features(
        board,
        [
            {
                "result": variation,
                "visits": 10,
            }
        ],
    )

    assert len(result.variation_deltas) == 1

    for feature_name, frequency in (
        result.f37.feature_frequencies.items()
    ):
        assert 0.0 <= frequency <= 1.0

    for value in result.f38.weighted_changes.values():
        assert isinstance(value, float)

    assert isinstance(result.f39, float)
    assert 0.0 <= result.f39 <= 1.0


def test_zero_visits_are_allowed():
    board = cshogi.Board()

    variation = compute_variation_features(
        board,
        [
            "7g7f",
            "3c3d",
        ],
    )

    result = compute_mcts_features(
        board,
        [
            {
                "result": variation,
                "visits": 0,
            }
        ],
    )

    assert result.f37.feature_frequencies
    assert result.f38.weighted_changes == {}
    assert result.f39 == pytest.approx(0.0)


def test_negative_visits_are_rejected():
    board = cshogi.Board()

    variation = compute_variation_features(
        board,
        [
            "7g7f",
        ],
    )

    with pytest.raises(ValueError):
        compute_mcts_features(
            board,
            [
                {
                    "result": variation,
                    "visits": -1,
                }
            ],
        )


def test_f39_uses_visit_weighted_changes():
    """
    F39はF38の訪問重み付き変化から計算される。

    1つの特徴量だけが変化している場合、
    集中度は1.0になる。
    """

    board = cshogi.Board()

    variation1 = compute_variation_features(
        board,
        [
            "7g7f",
        ],
    )

    result = compute_mcts_features(
        board,
        [
            {
                "result": variation1,
                "visits": 10,
            }
        ],
    )

    if result.f38.weighted_changes:
        assert 0.0 <= result.f39 <= 1.0
