import cshogi

from features.activity import BLACK, WHITE
from features.defensive_placement import (
    direction_distribution,
    direction_balance_score,
    defensive_proximity_score,
    f27_defensive_placement,
    f27_defensive_placement_change,
    gold_silver_placement_score,
)
from features.transition import apply_usi_move


def test_f27_initial_position_is_valid():
    board = cshogi.Board()

    black = f27_defensive_placement(board, BLACK)
    white = f27_defensive_placement(board, WHITE)

    assert 0.0 <= black.score <= 1.0
    assert 0.0 <= white.score <= 1.0

    assert black.proximity >= 0.0
    assert white.proximity >= 0.0

    assert black.gold_silver >= 0.0
    assert white.gold_silver >= 0.0

    assert 0.0 <= black.direction_balance <= 1.0
    assert 0.0 <= white.direction_balance <= 1.0


def test_f27_direction_distribution_has_eight_directions():
    board = cshogi.Board()

    black = direction_distribution(board, BLACK)
    white = direction_distribution(board, WHITE)

    expected = {
        "forward",
        "backward",
        "left",
        "right",
        "forward_left",
        "forward_right",
        "backward_left",
        "backward_right",
    }

    assert set(black.keys()) == expected
    assert set(white.keys()) == expected


def test_f27_direction_balance_is_normalized():
    board = cshogi.Board()

    black = direction_balance_score(board, BLACK)
    white = direction_balance_score(board, WHITE)

    assert 0.0 <= black <= 1.0
    assert 0.0 <= white <= 1.0


def test_f27_components_are_non_negative():
    board = cshogi.Board()

    for color in (BLACK, WHITE):
        assert defensive_proximity_score(board, color) >= 0.0
        assert gold_silver_placement_score(board, color) >= 0.0


def test_f27_change_is_zero_for_same_position():
    board = cshogi.Board()

    change = f27_defensive_placement_change(
        board,
        board.copy(),
    )

    assert abs(change.black) < 1e-12
    assert abs(change.white) < 1e-12
    assert abs(change.difference) < 1e-12


def test_f27_changes_after_move():
    board = cshogi.Board()
    after = apply_usi_move(board, "7g7f")

    change = f27_defensive_placement_change(
        board,
        after,
    )

    assert (
        change.black != 0.0
        or change.white != 0.0
    )
