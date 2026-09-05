import cshogi

from features.activity import BLACK, WHITE
from features.response import (
    f25_response,
    f25_response_change,
    f25_response_change_both,
)
from features.transition import apply_usi_move


def test_f25_initial_position_is_valid():
    board = cshogi.Board()

    black = f25_response(board, BLACK)
    white = f25_response(board, WHITE)

    # 初期局面では相手から直接攻撃されている駒がない。
    assert black.legal_defense == 0
    assert white.legal_defense == 0

    # 初期局面でも玉の逃走手数は存在する。
    assert black.king_escape >= 0
    assert white.king_escape >= 0

    assert black.counter_attack >= 0
    assert white.counter_attack >= 0

    assert 0.0 <= black.score <= 1.0
    assert 0.0 <= white.score <= 1.0


def test_f25_is_independent_of_board_turn():
    board = cshogi.Board()

    black_before = f25_response(board, BLACK)
    white_before = f25_response(board, WHITE)

    board.turn = cshogi.WHITE

    black_after = f25_response(board, BLACK)
    white_after = f25_response(board, WHITE)

    assert black_before == black_after
    assert white_before == white_after


def test_f25_change_is_zero_for_same_position():
    board = cshogi.Board()

    black = f25_response_change(
        board,
        board.copy(),
        BLACK,
    )

    white = f25_response_change(
        board,
        board.copy(),
        WHITE,
    )

    assert black.legal_defense == 0
    assert black.king_escape == 0
    assert black.counter_attack == 0
    assert black.score == 0

    assert white.legal_defense == 0
    assert white.king_escape == 0
    assert white.counter_attack == 0
    assert white.score == 0


def test_f25_change_after_move():
    board = cshogi.Board()

    for move in [
        "7g7f",
        "3c3d",
        "2g2f",
        "8c8d",
    ]:
        board = apply_usi_move(board, move)

    after = apply_usi_move(board, "6i7h")

    change = f25_response_change_both(
        board,
        after,
    )

    assert (
        change.black != 0
        or change.white != 0
    )


def test_f25_change_contains_component_changes():
    board = cshogi.Board()

    for move in [
        "7g7f",
        "3c3d",
        "2g2f",
        "8c8d",
    ]:
        board = apply_usi_move(board, move)

    after = apply_usi_move(board, "6i7h")

    change = f25_response_change(
        board,
        after,
        BLACK,
    )

    assert isinstance(change.legal_defense, float)
    assert isinstance(change.king_escape, float)
    assert isinstance(change.counter_attack, float)
    assert isinstance(change.score, float)

    assert change.score != 0.0
