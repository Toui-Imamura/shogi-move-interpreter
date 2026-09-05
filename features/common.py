"""
features/common.py

F01～F40で共通して利用する将棋盤面の基本処理。

使用ライブラリ:
    cshogi 1.0.4

設計方針:
    - cshogi固有のAPIをこのファイルに集約する
    - 他のfeature実装から直接cshogi内部表現を扱わない
    - Black / Whiteを明確に分離する
    - 駒価値・座標・持ち駒などの基本情報を統一する
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import cshogi


# ============================================================
# プレイヤー
# ============================================================

BLACK = cshogi.BLACK
WHITE = cshogi.WHITE


def opponent(color: int) -> int:
    """
    相手側の色を返す。
    """
    if color == BLACK:
        return WHITE
    if color == WHITE:
        return BLACK
    raise ValueError(f"invalid color: {color}")


# ============================================================
# 駒価値
# ============================================================

# 研究仕様書に合わせた駒価値。
#
# 歩      1
# 香      3
# 桂      3
# 銀      5
# 金      6
# 角      8
# 飛      9
# と金    9
# 成香    9
# 成桂    9
# 成銀    9
# 馬     11
# 龍     12
#
# 玉には通常の駒得評価を与えない。
PIECE_VALUES: Dict[int, float] = {
    cshogi.PAWN: 1.0,
    cshogi.LANCE: 3.0,
    cshogi.KNIGHT: 3.0,
    cshogi.SILVER: 5.0,
    cshogi.GOLD: 6.0,
    cshogi.BISHOP: 8.0,
    cshogi.ROOK: 9.0,
    cshogi.KING: 0.0,
    cshogi.PROM_PAWN: 9.0,
    cshogi.PROM_LANCE: 9.0,
    cshogi.PROM_KNIGHT: 9.0,
    cshogi.PROM_SILVER: 9.0,
    cshogi.PROM_BISHOP: 11.0,
    cshogi.PROM_ROOK: 12.0,
}


# ============================================================
# 駒種の分類
# ============================================================

MINOR_PIECES = {
    cshogi.PAWN,
    cshogi.LANCE,
    cshogi.KNIGHT,
    cshogi.SILVER,
    cshogi.GOLD,
    cshogi.PROM_PAWN,
    cshogi.PROM_LANCE,
    cshogi.PROM_KNIGHT,
    cshogi.PROM_SILVER,
}

MAJOR_PIECES = {
    cshogi.BISHOP,
    cshogi.ROOK,
    cshogi.PROM_BISHOP,
    cshogi.PROM_ROOK,
}


# 成駒を元の駒種へ戻すための対応。
PROMOTED_TO_BASE: Dict[int, int] = {
    cshogi.PROM_PAWN: cshogi.PAWN,
    cshogi.PROM_LANCE: cshogi.LANCE,
    cshogi.PROM_KNIGHT: cshogi.KNIGHT,
    cshogi.PROM_SILVER: cshogi.SILVER,
    cshogi.PROM_BISHOP: cshogi.BISHOP,
    cshogi.PROM_ROOK: cshogi.ROOK,
}


def base_piece_type(piece_type: int) -> int:
    """
    成駒を元の駒種に戻す。

    例:
        PROM_PAWN   -> PAWN
        PROM_BISHOP -> BISHOP
        ROOK        -> ROOK
    """
    return PROMOTED_TO_BASE.get(piece_type, piece_type)


def piece_value(piece_type: int) -> float:
    """
    駒種から研究仕様書上の評価値を取得する。
    """
    try:
        return PIECE_VALUES[piece_type]
    except KeyError as exc:
        raise ValueError(
            f"unsupported piece type: {piece_type}"
        ) from exc


# ============================================================
# 盤面上の駒情報
# ============================================================

@dataclass(frozen=True)
class PieceInfo:
    """
    盤上の1駒を表す。

    square:
        cshogiの盤面インデックス 0～80

    color:
        cshogi.BLACK / cshogi.WHITE

    piece_type:
        cshogiの駒種定数

    base_type:
        成駒であれば成る前の駒種

    value:
        研究仕様書上の駒価値
    """

    square: int
    color: int
    piece_type: int
    base_type: int
    value: float


# ============================================================
# 盤面関連
# ============================================================

def iter_pieces(board: cshogi.Board) -> Iterator[PieceInfo]:
    """
    盤上に存在する全駒を列挙する。

    cshogi.Board.pieces は81マス分の駒情報を返す。
    """
    pieces = board.pieces

    if len(pieces) != 81:
        raise ValueError(
            f"unexpected board size: {len(pieces)}"
        )

    for square, piece in enumerate(pieces):
        if piece == cshogi.NONE:
            continue

        # cshogiのpiece integerから駒種を取得する。
        piece_type = board.piece_type(square)

        # cshogiでは駒の所有者をpiece値から判定する。
        color = piece_color(piece)

        yield PieceInfo(
            square=square,
            color=color,
            piece_type=piece_type,
            base_type=base_piece_type(piece_type),
            value=piece_value(piece_type),
        )


def board_pieces(
    board: cshogi.Board,
    color: Optional[int] = None,
) -> List[PieceInfo]:
    """
    盤上の駒をリストとして取得する。

    colorを指定した場合、その側の駒だけを返す。
    """
    pieces = list(iter_pieces(board))

    if color is None:
        return pieces

    if color not in (BLACK, WHITE):
        raise ValueError(f"invalid color: {color}")

    return [piece for piece in pieces if piece.color == color]


def piece_color(piece: int) -> int:
    """
    cshogiのpiece値から所有者を取得する。

    cshogiではBLACK側とWHITE側でpiece定数が分かれている。
    """
    if cshogi.BPAWN <= piece <= cshogi.BPROM_ROOK:
        return BLACK

    if cshogi.WPAWN <= piece <= cshogi.WPROM_ROOK:
        return WHITE

    raise ValueError(f"unsupported piece value: {piece}")


# ============================================================
# 駒数
# ============================================================

def piece_counts(
    board: cshogi.Board,
    color: int,
    *,
    use_base_type: bool = False,
) -> Dict[int, int]:
    """
    盤上の駒数を駒種ごとに数える。

    use_base_type=True:
        成駒を元の駒種として集計する。

    例:
        と金 -> 歩
    """
    counts: Dict[int, int] = {}

    for piece in board_pieces(board, color):
        piece_type = (
            piece.base_type
            if use_base_type
            else piece.piece_type
        )

        counts[piece_type] = counts.get(piece_type, 0) + 1

    return counts


def count_piece(
    board: cshogi.Board,
    color: int,
    piece_type: int,
) -> int:
    """
    指定した駒種の盤上の枚数を取得する。
    """
    return sum(
        1
        for piece in board_pieces(board, color)
        if piece.piece_type == piece_type
    )


# ============================================================
# 持ち駒
# ============================================================

HAND_PIECE_TYPES = (
    cshogi.PAWN,
    cshogi.LANCE,
    cshogi.KNIGHT,
    cshogi.SILVER,
    cshogi.GOLD,
    cshogi.BISHOP,
    cshogi.ROOK,
)


def hand_counts(board: cshogi.Board, color: int) -> Dict[int, int]:
    """
    持ち駒を駒種ごとの辞書として取得する。

    cshogi.Board.pieces_in_hand は
        BLACK
        WHITE
    の順で7種類の駒数を保持する。
    """
    if color not in (BLACK, WHITE):
        raise ValueError(f"invalid color: {color}")

    hand = board.pieces_in_hand[color]

    if len(hand) != 7:
        raise ValueError(
            f"unexpected hand size: {len(hand)}"
        )

    return {
        piece_type: int(count)
        for piece_type, count in zip(HAND_PIECE_TYPES, hand)
    }


def hand_value(board: cshogi.Board, color: int) -> float:
    """
    持ち駒の価値合計を計算する。

    F02で使用する。
    """
    counts = hand_counts(board, color)

    return sum(
        piece_value(piece_type) * count
        for piece_type, count in counts.items()
    )


# ============================================================
# 盤上の駒価値
# ============================================================

def board_material(board: cshogi.Board, color: int) -> float:
    """
    盤上に存在する駒の価値合計。

    F01で使用する。
    """
    return sum(
        piece.value
        for piece in board_pieces(board, color)
    )


def total_material(board: cshogi.Board, color: int) -> float:
    """
    盤上 + 持ち駒の価値合計。
    """
    return (
        board_material(board, color)
        + hand_value(board, color)
    )


# ============================================================
# 玉
# ============================================================

def king_square(board: cshogi.Board, color: int) -> int:
    """
    指定した側の玉位置を返す。
    """
    if color not in (BLACK, WHITE):
        raise ValueError(f"invalid color: {color}")

    square = board.king_square(color)

    if square < 0 or square >= 81:
        raise ValueError(
            f"invalid king square: {square}"
        )

    return square


# ============================================================
# 座標
# ============================================================

def square_to_file_rank(square: int) -> Tuple[int, int]:
    """
    cshogiのsquare indexを

        file, rank

    に変換する。

    cshogiはApery形式の0～80インデックスを使用する。
    """
    if not 0 <= square < 81:
        raise ValueError(f"invalid square: {square}")

    file_index, rank_index = divmod(square, 9)

    return file_index, rank_index


def file_rank_to_square(file_index: int, rank_index: int) -> int:
    """
    file, rankからcshogiのsquare indexへ変換する。
    """
    if not 0 <= file_index < 9:
        raise ValueError(f"invalid file: {file_index}")

    if not 0 <= rank_index < 9:
        raise ValueError(f"invalid rank: {rank_index}")

    return file_index * 9 + rank_index


def square_distance(square_a: int, square_b: int) -> int:
    """
    2マス間のChebyshev距離を計算する。
    """
    file_a, rank_a = square_to_file_rank(square_a)
    file_b, rank_b = square_to_file_rank(square_b)

    return max(
        abs(file_a - file_b),
        abs(rank_a - rank_b),
    )


# ============================================================
# 合法手
# ============================================================

def legal_moves(board: cshogi.Board) -> List[int]:
    """
    現局面の合法手をリストとして取得する。

    F07などで使用する。
    """
    return list(board.legal_moves)


def legal_move_count(board: cshogi.Board) -> int:
    """
    現局面の合法手数。
    """
    return len(legal_moves(board))


# ============================================================
# 指し手適用
# ============================================================

def apply_move(
    board: cshogi.Board,
    move: int,
) -> cshogi.Board:
    """
    元のboardを変更せず、move後の盤面コピーを返す。

    MCTSやF03以降の変化量計算で使用する。
    """
    if not board.is_legal(move):
        raise ValueError(
            f"illegal move: {move}"
        )

    next_board = board.copy()
    next_board.push(move)

    return next_board


# ============================================================
# 正規化
# ============================================================

def tanh_normalize(value: float, scale: float) -> float:
    """
    連続値を[-1, 1]へ正規化する。

    仕様:
        tanh(x / c)
    """
    if scale <= 0:
        raise ValueError(
            f"scale must be positive: {scale}"
        )

    import math

    return math.tanh(value / scale)


def signed_difference(
    black_value: float,
    white_value: float,
) -> float:
    """
    Black - White。
    """
    return black_value - white_value


def mean_value(
    black_value: float,
    white_value: float,
) -> float:
    """
    (Black + White) / 2
    """
    return (black_value + white_value) / 2.0


# ============================================================
# デバッグ用
# ============================================================

def board_summary(board: cshogi.Board) -> Dict[str, object]:
    """
    現局面の基本情報をまとめて返す。

    テスト・デバッグ用。
    """
    return {
        "turn": board.turn,
        "move_number": board.move_number,
        "sfen": board.sfen(),
        "black_king": king_square(board, BLACK),
        "white_king": king_square(board, WHITE),
        "black_board_material": board_material(board, BLACK),
        "white_board_material": board_material(board, WHITE),
        "black_hand_value": hand_value(board, BLACK),
        "white_hand_value": hand_value(board, WHITE),
        "legal_move_count": legal_move_count(board),
    }
