"""
tests/test_pawn_structure.py

F31: 歩の連結性のテスト
"""

import cshogi

from features.pawn_structure import (
    _is_connected_pawn,
    connected_pawn_count,
    connected_pawn_pairs,
    f31_pawn_connectivity,
    f31_pawn_connectivity_change,
    pawn_connectivity_score,
    pawn_squares,
)


# ============================================================================
# Helper
# ============================================================================


def apply_moves(
    board: cshogi.Board,
    moves: list[str],
) -> cshogi.Board:
    """
    USI形式の指し手を順番に適用する。
    """

    for move_usi in moves:
        move = board.move_from_usi(
            move_usi
        )

        assert board.is_legal(
            move
        ), f"Illegal move: {move_usi}"

        board.push(move)

    return board


def square(
    usi: str,
) -> int:
    """
    USI座標からcshogi square indexを取得する。
    """

    return cshogi.SQUARE_NAMES.index(
        usi
    )


# ============================================================================
# Basic tests
# ============================================================================


def test_initial_pawn_count():
    board = cshogi.Board()

    black_pawns = pawn_squares(
        board,
        cshogi.BLACK,
    )

    white_pawns = pawn_squares(
        board,
        cshogi.WHITE,
    )

    assert len(black_pawns) == 9
    assert len(white_pawns) == 9


def test_initial_connectivity_is_zero():
    """
    初期局面では歩が同じ段に横並びなので、
    F31の前方斜め連結には該当しない。
    """

    board = cshogi.Board()

    assert (
        connected_pawn_count(
            board,
            cshogi.BLACK,
        )
        == 0
    )

    assert (
        connected_pawn_count(
            board,
            cshogi.WHITE,
        )
        == 0
    )


def test_initial_connectivity_score():
    board = cshogi.Board()

    assert (
        pawn_connectivity_score(
            board,
            cshogi.BLACK,
        )
        == 0.0
    )

    assert (
        pawn_connectivity_score(
            board,
            cshogi.WHITE,
        )
        == 0.0
    )


# ============================================================================
# Geometry tests
# ============================================================================


def test_forward_diagonal_is_connected_for_black():
    """
    黒:

        7g
         \
          6f

    は前方斜めなので連結。
    """

    assert _is_connected_pawn(
        square("7g"),
        square("6f"),
        cshogi.BLACK,
    )


def test_forward_diagonal_is_connected_for_white():
    """
    白:

        7c
         \
          6d

    は前方斜めなので連結。
    """

    assert _is_connected_pawn(
        square("7c"),
        square("6d"),
        cshogi.WHITE,
    )


def test_same_rank_adjacent_pawns_are_not_connected():
    """
    7fと6fは隣接する筋だが同じ段なので、
    F31では連結とは扱わない。
    """

    assert not _is_connected_pawn(
        square("7f"),
        square("6f"),
        cshogi.BLACK,
    )


def test_backward_diagonal_is_not_connected_for_black():
    """
    黒にとって、

        7f
         \
          6g

    は後方斜めなので連結ではない。
    """

    assert not _is_connected_pawn(
        square("7f"),
        square("6g"),
        cshogi.BLACK,
    )


def test_backward_diagonal_is_not_connected_for_white():
    """
    白にとって、

        7f
         \
          6e

    は後方斜めなので連結ではない。
    """

    assert not _is_connected_pawn(
        square("7f"),
        square("6e"),
        cshogi.WHITE,
    )


def test_non_adjacent_file_is_not_connected():
    """
    筋が2つ以上離れている場合は連結ではない。
    """

    assert not _is_connected_pawn(
        square("7g"),
        square("5f"),
        cshogi.BLACK,
    )


# ============================================================================
# Actual board tests
# ============================================================================


def test_adjacent_forward_pawns_create_connectivity():
    """
    黒:

        7g7f
        6g6f

    により、

        7g
         \
          6f

    という連結関係が発生する。

    さらに歩を進めることで、
    別の連結関係も形成される。
    """

    board = cshogi.Board()

    moves = [
        "7g7f",
        "3c3d",
        "6g6f",
        "8c8d",
        "7f7e",
        "4c4d",
        "6f6e",
        "9c9d",
        "6e6d",
    ]

    board = apply_moves(
        board,
        moves,
    )

    connected = connected_pawn_count(
        board,
        cshogi.BLACK,
    )

    pairs = connected_pawn_pairs(
        board,
        cshogi.BLACK,
    )

    score = pawn_connectivity_score(
        board,
        cshogi.BLACK,
    )

    assert connected > 0

    assert len(pairs) > 0

    assert 0.0 <= score <= 1.0


# ============================================================================
# Feature tests
# ============================================================================


def test_f31_feature():
    board = cshogi.Board()

    moves = [
        "7g7f",
        "3c3d",
        "6g6f",
        "8c8d",
        "7f7e",
        "4c4d",
        "6f6e",
        "9c9d",
        "6e6d",
    ]

    board = apply_moves(
        board,
        moves,
    )

    feature = f31_pawn_connectivity(
        board,
        cshogi.BLACK,
    )

    assert feature.total_pawns == 9

    assert feature.connected_pawns > 0

    assert (
        0.0
        <= feature.connected_ratio
        <= 1.0
    )

    assert (
        feature.score
        == feature.connected_ratio
    )


def test_f31_change():
    before = cshogi.Board()

    after = cshogi.Board()

    moves = [
        "7g7f",
        "3c3d",
        "6g6f",
    ]

    after = apply_moves(
        after,
        moves,
    )

    change = f31_pawn_connectivity_change(
        before,
        after,
    )

    assert isinstance(
        change.black,
        float,
    )

    assert isinstance(
        change.white,
        float,
    )

    assert isinstance(
        change.difference,
        float,
    )

    assert (
        -1.0
        <= change.black
        <= 1.0
    )

    assert (
        -1.0
        <= change.white
        <= 1.0
    )

    assert (
        -1.0
        <= change.difference
        <= 1.0
    )