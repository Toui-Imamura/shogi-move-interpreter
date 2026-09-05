import cshogi

from features.activity import BLACK, WHITE
from features.king_safety import (
    f24_king_safety_change,
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


def test_f24_unchanged_position():
    board = cshogi.Board()

    result = f24_king_safety_change(
        board,
        board.copy(),
    )

    assert result.black == 0.0
    assert result.white == 0.0
    assert result.difference == 0.0


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


def test_f24_after_move():
    board = cshogi.Board()

    move = board.move_from_usi("7g7f")

    after = board.copy()
    after.push(move)

    result = f24_king_safety_change(
        board,
        after,
    )

    assert isinstance(result.black, float)
    assert isinstance(result.white, float)
    assert isinstance(result.difference, float)
