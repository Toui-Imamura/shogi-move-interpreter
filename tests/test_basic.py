import cshogi

from features.basic import (
    f01_material,
    f02_hand_material,
    f03_exchange,
    f04_major_balance,
    f05_minor_composition,
)


def test_f01_initial_position():
    board = cshogi.Board()

    result = f01_material(board)

    assert result.black == result.white
    assert result.difference == 0.0


def test_f02_initial_position():
    board = cshogi.Board()

    result = f02_hand_material(board)

    assert result.black == 0.0
    assert result.white == 0.0
    assert result.difference == 0.0


def test_f04_initial_position():
    board = cshogi.Board()

    result = f04_major_balance(board)

    assert result.black == result.white
    assert result.difference == 0.0

    assert result.black_count == 2
    assert result.white_count == 2


def test_f05_initial_position():
    board = cshogi.Board()

    result = f05_minor_composition(board)

    assert result.difference[cshogi.PAWN] == 0
    assert result.difference[cshogi.LANCE] == 0
    assert result.difference[cshogi.KNIGHT] == 0
    assert result.difference[cshogi.SILVER] == 0
    assert result.difference[cshogi.GOLD] == 0


def test_f03_capture():
    board = cshogi.Board()

    capture_move = None

    for _ in range(30):
        moves = list(board.legal_moves)

        for move in moves:
            if cshogi.move_cap(move) != cshogi.NONE:
                capture_move = move
                break

        if capture_move is not None:
            break

        board.push(moves[0])

    assert capture_move is not None

    before = board.copy()

    after = board.copy()
    after.push(capture_move)

    result = f03_exchange(
        before,
        after,
        capture_move,
    )

    assert result.exchange_occurred == 1
    assert result.captured_piece != cshogi.NONE
    assert result.captured_piece_value > 0.0
