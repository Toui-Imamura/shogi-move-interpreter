import cshogi

from features.activity import BLACK, WHITE
from features.king_safety import (
    f26_king_safety,
    king_attackers,
    king_escape_count,
    own_king_control,
    enemy_king_control,
)


def test_f26_initial_position():
    board = cshogi.Board()

    result = f26_king_safety(board)

    assert isinstance(result.black, float)
    assert isinstance(result.white, float)

    assert -1.0 <= result.black <= 1.0
    assert -1.0 <= result.white <= 1.0


def test_f26_initial_components_are_non_negative():
    board = cshogi.Board()

    result = f26_king_safety(board)

    assert result.black_own_control >= 0
    assert result.black_enemy_control >= 0
    assert result.black_escape >= 0
    assert result.black_attackers >= 0

    assert result.white_own_control >= 0
    assert result.white_enemy_control >= 0
    assert result.white_escape >= 0
    assert result.white_attackers >= 0


def test_f26_unchanged_position():
    board = cshogi.Board()

    result = f26_king_safety(board)

    assert result.difference == result.black - result.white


def test_king_escape_count_initial_position():
    board = cshogi.Board()

    black_escape = king_escape_count(
        board,
        BLACK,
    )

    white_escape = king_escape_count(
        board,
        WHITE,
    )

    assert black_escape >= 0
    assert white_escape >= 0


def test_king_control_non_negative():
    board = cshogi.Board()

    assert own_king_control(board, BLACK) >= 0
    assert own_king_control(board, WHITE) >= 0

    assert enemy_king_control(board, BLACK) >= 0
    assert enemy_king_control(board, WHITE) >= 0


def test_king_attackers_non_negative():
    board = cshogi.Board()

    assert king_attackers(board, BLACK) >= 0
    assert king_attackers(board, WHITE) >= 0
