import cshogi
import pytest

from features.transition import (
    apply_move,
    apply_usi_move,
    make_transition,
    make_usi_variation,
    make_variation,
    position_at,
)


def test_apply_move_does_not_modify_original_board():
    """apply_moveは元のBoardを変更しない。"""
    board = cshogi.Board()

    move = board.move_from_usi("7g7f")

    before_sfen = board.sfen()

    next_board = apply_move(board, move)

    assert board.sfen() == before_sfen
    assert next_board.sfen() != before_sfen


def test_apply_usi_move():
    """USI形式の指し手を正しく適用できる。"""
    board = cshogi.Board()

    next_board = apply_usi_move(board, "7g7f")

    assert board.piece(cshogi.SQUARE_NAMES.index("7g")) != 0
    assert next_board.piece(cshogi.SQUARE_NAMES.index("7g")) == 0
    assert next_board.piece(cshogi.SQUARE_NAMES.index("7f")) != 0


def test_make_transition():
    """S0→S1の遷移を作成できる。"""
    board = cshogi.Board()

    move = board.move_from_usi("7g7f")

    transition = make_transition(board, move)

    assert transition.before.sfen() == board.sfen()
    assert transition.move == move
    assert transition.after.sfen() != board.sfen()


def test_make_variation():
    """複数手の変化列S0→S1→...→Svを作成できる。"""
    board = cshogi.Board()

    board_1 = board.copy()
    move_1 = board_1.move_from_usi("7g7f")
    board_1.push(move_1)

    board_2 = board_1.copy()
    move_2 = board_2.move_from_usi("3c3d")
    board_2.push(move_2)

    board_3 = board_2.copy()
    move_3 = board_3.move_from_usi("2g2f")

    moves = [
        move_1,
        move_2,
        move_3,
    ]

    variation = make_variation(board, moves)

    assert variation.length == 3
    assert len(variation.positions) == 4

    assert variation.positions[0].sfen() == board.sfen()
    assert variation.final.sfen() == variation.positions[3].sfen()


def test_make_usi_variation():
    """USI形式の指し手列からVariationを作成できる。"""
    board = cshogi.Board()

    usi_moves = [
        "7g7f",
        "3c3d",
        "2g2f",
    ]

    variation = make_usi_variation(board, usi_moves)

    assert variation.length == 3
    assert len(variation.positions) == 4


def test_position_at():
    """Variationから任意の手数の局面を取得できる。"""
    board = cshogi.Board()

    variation = make_usi_variation(
        board,
        [
            "7g7f",
            "3c3d",
        ],
    )

    assert position_at(variation, 0).sfen() == board.sfen()

    assert position_at(variation, 1).piece(
        cshogi.SQUARE_NAMES.index("7f")
    ) != 0

    assert position_at(variation, 2).piece(
        cshogi.SQUARE_NAMES.index("3d")
    ) != 0


def test_illegal_move_raises_error():
    """不正な指し手を適用するとValueErrorになる。"""
    board = cshogi.Board()

    # 初期局面で7g7eは歩の2マス前進なので不合法。
    move = board.move_from_usi("7g7e")

    with pytest.raises(ValueError):
        apply_move(board, move)
