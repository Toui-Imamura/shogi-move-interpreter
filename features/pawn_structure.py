"""
features/pawn_structure.py

F31: 歩の連結性

自軍の歩について、

    「隣接する筋に存在し、かつ自軍の前進方向に1段進んだ位置」

に存在する歩との関係を連結構造として評価する。

F31では、cshogiの内部square indexではなく、
cshogi.SQUARE_NAMESを利用して将棋盤上の座標関係を判定する。

黒:
    rankが小さくなる方向へ進む

    7g -> 6f
    7e -> 6d

白:
    rankが大きくなる方向へ進む

    7c -> 6d
    7e -> 6f
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple

import cshogi

from .common import (
    BLACK,
    WHITE,
    board_pieces,
)


# ============================================================================
# Pawn extraction
# ============================================================================


def pawn_squares(
    board: cshogi.Board,
    color: int,
) -> List[int]:
    """
    指定した色の盤上の歩のsquare一覧を返す。

    board_pieces() は PieceInfo を返すため、
    PieceInfo.color / PieceInfo.base_type を直接利用する。
    """

    squares: List[int] = []

    for piece in board_pieces(board):
        # PieceInfo.color
        if piece.color != color:
            continue

        # PieceInfo.base_type
        if piece.base_type != cshogi.PAWN:
            continue

        squares.append(piece.square)

    return squares


# ============================================================================
# Coordinate utilities
# ============================================================================


def _usi_position(
    square: int,
) -> Tuple[int, int]:
    """
    cshogiのsquare indexをUSI座標に変換する。

    例:
        7g -> (7, 7)
        6f -> (6, 6)
        6d -> (6, 4)

    Returns
    -------
    Tuple[int, int]
        (筋, 段)
    """

    usi = cshogi.SQUARE_NAMES[square]

    file_number = int(usi[0])

    rank_number = (
        ord(usi[1]) - ord("a") + 1
    )

    return file_number, rank_number


# ============================================================================
# Connection definition
# ============================================================================


def _is_connected_pawn(
    square: int,
    other_square: int,
    color: int,
) -> bool:
    """
    2枚の歩がF31における連結関係を形成しているか判定する。

    条件:

        1. 隣接する筋に存在する
        2. 自軍の前進方向に1段離れている

    黒:
        rankが1小さい位置が前方

        7g -> 6f
        7e -> 6d

    白:
        rankが1大きい位置が前方

        7c -> 6d
        7e -> 6f

    Parameters
    ----------
    square : int
        基準となる歩のsquare

    other_square : int
        相手側の歩のsquare

    color : int
        cshogi.BLACK または cshogi.WHITE

    Returns
    -------
    bool
        連結関係ならTrue
    """

    file_number, rank_number = _usi_position(
        square
    )

    other_file, other_rank = _usi_position(
        other_square
    )

    # ------------------------------------------------------------------
    # 筋が隣接している必要がある
    # ------------------------------------------------------------------

    if abs(
        file_number - other_file
    ) != 1:
        return False

    # ------------------------------------------------------------------
    # 黒
    # ------------------------------------------------------------------

    if color == BLACK:
        return (
            other_rank
            == rank_number - 1
        )

    # ------------------------------------------------------------------
    # 白
    # ------------------------------------------------------------------

    if color == WHITE:
        return (
            other_rank
            == rank_number + 1
        )

    raise ValueError(
        f"Invalid color: {color}"
    )


# ============================================================================
# Connected pawn pairs
# ============================================================================


def connected_pawn_pairs(
    board: cshogi.Board,
    color: int,
) -> Set[Tuple[int, int]]:
    """
    F31における歩の連結ペアを取得する。

    同じ2枚の歩について重複して数えない。

    Returns
    -------
    Set[Tuple[int, int]]
        連結している歩のsquare pair
    """

    pawns = pawn_squares(
        board,
        color,
    )

    pairs: Set[Tuple[int, int]] = set()

    for i, square in enumerate(pawns):
        for other_square in pawns[i + 1:]:

            # どちらを基準にしても、
            # 前方斜めの関係が成立する可能性がある。
            connected = (
                _is_connected_pawn(
                    square,
                    other_square,
                    color,
                )
                or
                _is_connected_pawn(
                    other_square,
                    square,
                    color,
                )
            )

            if not connected:
                continue

            # squareの大小で正規化して重複を防ぐ
            pair = tuple(
                sorted(
                    (
                        square,
                        other_square,
                    )
                )
            )

            pairs.add(pair)

    return pairs


# ============================================================================
# Connected pawn count
# ============================================================================


def connected_pawn_count(
    board: cshogi.Board,
    color: int,
) -> int:
    """
    F31において連結構造を形成している歩の枚数を返す。

    1つの連結ペアに含まれる歩は2枚として数える。

    例えば、

        7e
         \
          6d

    なら2枚として数える。
    """

    pairs = connected_pawn_pairs(
        board,
        color,
    )

    connected: Set[int] = set()

    for square_a, square_b in pairs:
        connected.add(square_a)
        connected.add(square_b)

    return len(connected)


# ============================================================================
# Connectivity score
# ============================================================================


def pawn_connectivity_score(
    board: cshogi.Board,
    color: int,
) -> float:
    """
    F31の歩の連結性スコアを計算する。

    定義:

        connected_pawns / total_pawns

    値域:

        0.0 <= score <= 1.0

    歩が存在しない場合は0.0を返す。
    """

    pawns = pawn_squares(
        board,
        color,
    )

    total_pawns = len(pawns)

    if total_pawns == 0:
        return 0.0

    connected = connected_pawn_count(
        board,
        color,
    )

    return connected / total_pawns


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass(frozen=True)
class F31PawnConnectivity:
    """
    F31: 歩の連結性
    """

    score: float
    connected_pawns: int
    total_pawns: int
    connected_ratio: float


@dataclass(frozen=True)
class F31PawnConnectivityChange:
    """
    F31の局面変化。

    before -> after

    の差分を保持する。
    """

    black: float
    white: float
    difference: float


# ============================================================================
# Feature extraction
# ============================================================================


def f31_pawn_connectivity(
    board: cshogi.Board,
    color: int,
) -> F31PawnConnectivity:
    """
    指定した色についてF31を抽出する。
    """

    pawns = pawn_squares(
        board,
        color,
    )

    total_pawns = len(pawns)

    connected = connected_pawn_count(
        board,
        color,
    )

    if total_pawns == 0:
        ratio = 0.0
    else:
        ratio = connected / total_pawns

    return F31PawnConnectivity(
        score=ratio,
        connected_pawns=connected,
        total_pawns=total_pawns,
        connected_ratio=ratio,
    )


# ============================================================================
# Feature change
# ============================================================================


def f31_pawn_connectivity_change(
    before: cshogi.Board,
    after: cshogi.Board,
) -> F31PawnConnectivityChange:
    """
    beforeからafterへのF31変化量を計算する。

        ΔF31_black
        ΔF31_white
        ΔF31_difference

    を返す。
    """

    before_black = f31_pawn_connectivity(
        before,
        BLACK,
    )

    after_black = f31_pawn_connectivity(
        after,
        BLACK,
    )

    before_white = f31_pawn_connectivity(
        before,
        WHITE,
    )

    after_white = f31_pawn_connectivity(
        after,
        WHITE,
    )

    black_change = (
        after_black.score
        - before_black.score
    )

    white_change = (
        after_white.score
        - before_white.score
    )

    difference_change = (
        black_change
        - white_change
    )

    return F31PawnConnectivityChange(
        black=black_change,
        white=white_change,
        difference=difference_change,
    )