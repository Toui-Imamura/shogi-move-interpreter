import cshogi

from features.common import BLACK, WHITE
from features.force_distribution import (
    f33_force_distribution,
    f33_force_distribution_change,
    f33_force_distribution_change_both,
)


def apply_moves(moves):
    board = cshogi.Board()

    for usi in moves:
        move = board.move_from_usi(usi)

        assert board.is_legal(move), (
            f"illegal move: {usi}"
        )

        board.push(move)

    return board


def test_initial_position_black():
    board = cshogi.Board()

    result = f33_force_distribution(
        board,
        BLACK,
    )

    assert 0.0 <= result.left <= 1.0
    assert 0.0 <= result.center <= 1.0
    assert 0.0 <= result.right <= 1.0
    assert 0.0 <= result.own_side <= 1.0
    assert 0.0 <= result.enemy_side <= 1.0

    assert -1.0 <= result.score <= 1.0


def test_initial_position_white():
    board = cshogi.Board()

    result = f33_force_distribution(
        board,
        WHITE,
    )

    assert 0.0 <= result.left <= 1.0
    assert 0.0 <= result.center <= 1.0
    assert 0.0 <= result.right <= 1.0
    assert 0.0 <= result.own_side <= 1.0
    assert 0.0 <= result.enemy_side <= 1.0

    assert -1.0 <= result.score <= 1.0


def test_distribution_sums_to_one():
    board = cshogi.Board()

    result = f33_force_distribution(
        board,
        BLACK,
    )

    horizontal_total = (
        result.left
        + result.center
        + result.right
    )

    vertical_total = (
        result.own_side
        + result.enemy_side
    )

    assert abs(horizontal_total - 1.0) < 1e-9
    assert abs(vertical_total - 1.0) < 1e-9


def test_vector():
    board = cshogi.Board()

    result = f33_force_distribution(
        board,
        BLACK,
    )

    assert len(result.vector) == 5

    assert result.vector == (
        result.left,
        result.center,
        result.right,
        result.own_side,
        result.enemy_side,
    )


def test_actual_move_changes_distribution():
    before = cshogi.Board()

    after = apply_moves([
        "7g7f",
    ])

    result = f33_force_distribution_change(
        before,
        after,
        BLACK,
    )

    assert isinstance(result.score, float)

    assert -1.0 <= result.score <= 1.0


def test_change_same_position_is_zero():
    board = cshogi.Board()

    result = f33_force_distribution_change(
        board,
        board,
        BLACK,
    )

    assert abs(result.score) < 1e-9
    assert abs(result.left) < 1e-9
    assert abs(result.center) < 1e-9
    assert abs(result.right) < 1e-9
    assert abs(result.own_side) < 1e-9
    assert abs(result.enemy_side) < 1e-9


def test_black_and_white_change():
    before = cshogi.Board()

    after = apply_moves([
        "7g7f",
        "3c3d",
        "2g2f",
    ])

    result = f33_force_distribution_change_both(
        before,
        after,
    )

    assert isinstance(result.black.score, float)
    assert isinstance(result.white.score, float)

    assert len(result.difference) == 5


def test_invalid_color():
    board = cshogi.Board()

    try:
        f33_force_distribution(
            board,
            999,
        )
        assert False
    except ValueError:
        pass


def test_distribution_range_after_moves():
    board = apply_moves([
        "7g7f",
        "3c3d",
        "2g2f",
        "8c8d",
    ])

    result = f33_force_distribution(
        board,
        BLACK,
    )

    assert 0.0 <= result.left <= 1.0
    assert 0.0 <= result.center <= 1.0
    assert 0.0 <= result.right <= 1.0
    assert 0.0 <= result.own_side <= 1.0
    assert 0.0 <= result.enemy_side <= 1.0
    assert -1.0 <= result.score <= 1.0