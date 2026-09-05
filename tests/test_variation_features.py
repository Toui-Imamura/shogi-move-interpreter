import cshogi

from interpreter.variation_features import (
    compute_variation_features,
)


def test_compute_variation_features():
    board = cshogi.Board()

    moves = [
        "7g7f",
        "3c3d",
        "2g2f",
    ]

    result = compute_variation_features(
        board,
        moves,
    )

    # 局面数
    assert len(result.positions) == 4

    # 遷移数
    assert len(result.transition_deltas) == 3

    # F13 / F15
    assert "F13" in result.variation_deltas
    assert "F15" in result.variation_deltas

    # F34
    assert "F34_material" in result.variation_deltas
    assert "F34_attack" in result.variation_deltas
    assert "F34_defense" in result.variation_deltas
    assert "F34_activity" in result.variation_deltas
    assert "F34_formation" in result.variation_deltas

    # F35
    assert "F35" in result.variation_deltas
    assert 0.0 <= result.variation_deltas["F35"] <= 1.0

    # F36
    assert "F36" in result.variation_deltas
    assert 0.0 <= result.variation_deltas["F36"] <= 1.0

    # 各遷移
    for deltas in result.transition_deltas:
        assert isinstance(deltas, dict)
        assert "F01" in deltas
        assert "F33" in deltas

        # F13/F15はvariation全体で計算するため、
        # transitionには重複して入れない
        assert "F13" not in deltas
        assert "F15" not in deltas


def test_empty_variation():
    board = cshogi.Board()

    result = compute_variation_features(
        board,
        [],
    )

    assert len(result.positions) == 1
    assert len(result.transition_deltas) == 0
    assert result.variation_deltas == {}