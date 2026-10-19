import cshogi
import pytest

from features.square_importance import (
    build_square_effects,
    changed_square_effects,
    square_to_usi,
    top_important_squares,
)


def test_square_to_usi():
    assert square_to_usi(60) == "7g"
    assert square_to_usi(59) == "7f"


@pytest.mark.parametrize("square", [-1, 81])
def test_square_to_usi_rejects_invalid_square(square):
    with pytest.raises(ValueError):
        square_to_usi(square)


def test_all_81_squares_are_included():
    board = cshogi.Board()

    effects = build_square_effects(board)

    assert len(effects) == 81
    assert {item.square for item in effects} == set(range(81))


def test_empty_squares_are_included():
    board = cshogi.Board()

    effects = build_square_effects(board)
    empty_squares = [
        item for item in effects
        if item.piece == cshogi.NONE
    ]

    assert empty_squares


def test_initial_position_has_symmetric_control_counts():
    board = cshogi.Board()

    effects = build_square_effects(board)

    assert sum(item.black_attackers for item in effects) == sum(
        item.white_attackers for item in effects
    )


def test_same_position_has_no_changes():
    board = cshogi.Board()

    assert changed_square_effects(board, board) == ()


def test_move_changes_square_effects():
    before = cshogi.Board()
    after = cshogi.Board()
    after.push_usi("7g7f")

    changes = changed_square_effects(before, after)

    assert changes


def test_top_important_squares_returns_requested_count():
    board = cshogi.Board()

    result = top_important_squares(board, top_n=10)

    assert len(result) == 10
    assert all(0 <= item.square < 81 for item in result)
    assert list(result) == sorted(
        result,
        key=lambda item: (
            item.importance,
            item.contested,
            item.black_attackers + item.white_attackers,
            -item.square,
        ),
        reverse=True,
    )


def test_empty_square_only_ranking():
    board = cshogi.Board()

    result = top_important_squares(
        board,
        top_n=10,
        include_occupied=False,
    )

    assert len(result) == 10
    assert all(item.piece == cshogi.NONE for item in result)
