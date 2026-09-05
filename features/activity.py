"""
activity.py

F06～F10:
    F06 駒の利き
    F07 駒の移動可能性
    F08 重要地点への利き
    F09 遊び駒の改善
    F10 活動性の変化

cshogi 1.0.4 には「ある駒の利き」を直接取得する
APIが存在しないため、本ファイルではcshogiの盤面情報と
将棋の駒の移動規則から利きを計算する。

座標:
    cshogi.SQUARE_NAMES:
        1a ... 9i

    square = file * 9 + rank

    file:
        0 = 1筋
        8 = 9筋

    rank:
        0 = a
        8 = i

手番:
    Black:
        rank方向 -1

    White:
        rank方向 +1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import cshogi

from .common import (
    BLACK,
    WHITE,
    king_square,
    legal_moves,
    legal_move_count,
    tanh_normalize,
)


# ---------------------------------------------------------------------------
# Piece type constants
# ---------------------------------------------------------------------------

NONE = 0

PAWN = cshogi.PAWN
LANCE = cshogi.LANCE
KNIGHT = cshogi.KNIGHT
SILVER = cshogi.SILVER
BISHOP = cshogi.BISHOP
ROOK = cshogi.ROOK
GOLD = cshogi.GOLD
KING = cshogi.KING

PROM_PAWN = cshogi.PROM_PAWN
PROM_LANCE = cshogi.PROM_LANCE
PROM_KNIGHT = cshogi.PROM_KNIGHT
PROM_SILVER = cshogi.PROM_SILVER
PROM_BISHOP = cshogi.PROM_BISHOP
PROM_ROOK = cshogi.PROM_ROOK


# ---------------------------------------------------------------------------
# Piece / board helpers
# ---------------------------------------------------------------------------

def square_to_file_rank(square: int) -> Tuple[int, int]:
    """
    cshogiの81マス番号を(file, rank)へ変換する。

    file:
        0 = 1筋
        ...
        8 = 9筋

    rank:
        0 = a
        ...
        8 = i
    """
    if not 0 <= square < 81:
        raise ValueError(f"invalid square: {square}")

    return square // 9, square % 9


def file_rank_to_square(file: int, rank: int) -> Optional[int]:
    """(file, rank)をcshogiの81マス番号へ変換する。"""
    if not (0 <= file < 9 and 0 <= rank < 9):
        return None

    return file * 9 + rank


def piece_color(piece: int) -> Optional[int]:
    """
    駒の所有者を返す。

    Returns:
        BLACK
        WHITE
        None
    """
    if cshogi.BPAWN <= piece <= cshogi.BKING:
        return BLACK

    if cshogi.WPAWN <= piece <= cshogi.WKING:
        return WHITE

    return None


def opponent(color: int) -> int:
    """手番を反転する。"""
    if color == BLACK:
        return WHITE

    if color == WHITE:
        return BLACK

    raise ValueError(f"invalid color: {color}")


def forward_direction(color: int) -> int:
    """
    各陣営の前方向。

    初期局面:
        Blackの歩: 7g -> 7f
        Whiteの歩: 3c -> 3d
    """
    if color == BLACK:
        return -1

    if color == WHITE:
        return 1

    raise ValueError(f"invalid color: {color}")


def is_inside(file: int, rank: int) -> bool:
    return 0 <= file < 9 and 0 <= rank < 9


def step_square(
    square: int,
    df: int,
    dr: int,
) -> Optional[int]:
    """squareから(file方向, rank方向)に1マス移動する。"""
    file, rank = square_to_file_rank(square)

    return file_rank_to_square(
        file + df,
        rank + dr,
    )


# ---------------------------------------------------------------------------
# Attack generation
# ---------------------------------------------------------------------------

def _add_step_attack(
    board: cshogi.Board,
    square: int,
    color: int,
    df: int,
    dr: int,
    attacks: Set[int],
) -> None:
    """1マス移動する駒の利きを追加する。"""
    target = step_square(square, df, dr)

    if target is None:
        return

    target_piece = board.piece(target)

    # 自分の駒があるマスも「守っているマス」として利きに含める。
    # ただし、その先には進まない。
    attacks.add(target)

    if target_piece != 0:
        return


def _add_slider_attacks(
    board: cshogi.Board,
    square: int,
    directions: Iterable[Tuple[int, int]],
    attacks: Set[int],
) -> None:
    """飛車・角・香などの長距離駒の利きを追加する。"""
    for df, dr in directions:
        current = square

        while True:
            target = step_square(current, df, dr)

            if target is None:
                break

            attacks.add(target)

            # 駒に当たったら、その先には利かない。
            if board.piece(target) != 0:
                break

            current = target


def _gold_attacks(
    board: cshogi.Board,
    square: int,
    color: int,
    attacks: Set[int],
) -> None:
    f = forward_direction(color)

    directions = [
        (-1, f),
        (0, f),
        (1, f),
        (-1, 0),
        (1, 0),
        (0, -f),
    ]

    for df, dr in directions:
        _add_step_attack(board, square, color, df, dr, attacks)


def _silver_attacks(
    board: cshogi.Board,
    square: int,
    color: int,
    attacks: Set[int],
) -> None:
    f = forward_direction(color)

    directions = [
        (-1, f),
        (0, f),
        (1, f),
        (-1, -f),
        (1, -f),
    ]

    for df, dr in directions:
        _add_step_attack(board, square, color, df, dr, attacks)


def _king_attacks(
    board: cshogi.Board,
    square: int,
    color: int,
    attacks: Set[int],
) -> None:
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue

            _add_step_attack(
                board,
                square,
                color,
                df,
                dr,
                attacks,
            )


def _knight_attacks(
    board: cshogi.Board,
    square: int,
    color: int,
    attacks: Set[int],
) -> None:
    f = forward_direction(color)

    directions = [
        (-1, 2 * f),
        (1, 2 * f),
    ]

    for df, dr in directions:
        _add_step_attack(
            board,
            square,
            color,
            df,
            dr,
            attacks,
        )


def _bishop_attacks(
    board: cshogi.Board,
    square: int,
    attacks: Set[int],
) -> None:
    _add_slider_attacks(
        board,
        square,
        [
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ],
        attacks,
    )


def _rook_attacks(
    board: cshogi.Board,
    square: int,
    attacks: Set[int],
) -> None:
    _add_slider_attacks(
        board,
        square,
        [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ],
        attacks,
    )


def attacks_from_square(
    board: cshogi.Board,
    square: int,
) -> Set[int]:
    """
    指定マスにいる駒の利きを返す。

    注意:
        これは「合法手」ではなく、盤面上の駒の移動規則に
        基づく攻撃・防御可能マスを返す。

        王手回避などの合法性はここでは判定しない。
    """
    piece = board.piece(square)

    if piece == 0:
        return set()

    color = piece_color(piece)

    if color is None:
        return set()

    piece_type = board.piece_type(square)

    attacks: Set[int] = set()

    # 歩
    if piece_type == PAWN:
        f = forward_direction(color)
        _add_step_attack(
            board,
            square,
            color,
            0,
            f,
            attacks,
        )

    # 香
    elif piece_type == LANCE:
        f = forward_direction(color)

        _add_slider_attacks(
            board,
            square,
            [(0, f)],
            attacks,
        )

    # 桂
    elif piece_type == KNIGHT:
        _knight_attacks(
            board,
            square,
            color,
            attacks,
        )

    # 銀
    elif piece_type == SILVER:
        _silver_attacks(
            board,
            square,
            color,
            attacks,
        )

    # 角
    elif piece_type == BISHOP:
        _bishop_attacks(
            board,
            square,
            attacks,
        )

    # 飛
    elif piece_type == ROOK:
        _rook_attacks(
            board,
            square,
            attacks,
        )

    # 金
    elif piece_type == GOLD:
        _gold_attacks(
            board,
            square,
            color,
            attacks,
        )

    # 玉
    elif piece_type == KING:
        _king_attacks(
            board,
            square,
            color,
            attacks,
        )

    # 成歩・成香・成桂・成銀
    elif piece_type in (
        PROM_PAWN,
        PROM_LANCE,
        PROM_KNIGHT,
        PROM_SILVER,
    ):
        _gold_attacks(
            board,
            square,
            color,
            attacks,
        )

    # 馬
    elif piece_type == PROM_BISHOP:
        _bishop_attacks(
            board,
            square,
            attacks,
        )

        for df, dr in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]:
            _add_step_attack(
                board,
                square,
                color,
                df,
                dr,
                attacks,
            )

    # 龍
    elif piece_type == PROM_ROOK:
        _rook_attacks(
            board,
            square,
            attacks,
        )

        for df, dr in [
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            _add_step_attack(
                board,
                square,
                color,
                df,
                dr,
                attacks,
            )

    return attacks


def all_attacks(
    board: cshogi.Board,
    color: int,
) -> Dict[int, Set[int]]:
    """
    指定陣営の全駒について利きを計算する。

    Returns:
        {
            駒の位置: {利いているマス, ...},
            ...
        }
    """
    result: Dict[int, Set[int]] = {}

    for square in range(81):
        piece = board.piece(square)

        if piece == 0:
            continue

        if piece_color(piece) != color:
            continue

        result[square] = attacks_from_square(
            board,
            square,
        )

    return result


def controlled_squares(
    board: cshogi.Board,
    color: int,
) -> Set[int]:
    """指定陣営が利かせているマスの集合。"""
    result: Set[int] = set()

    for attacks in all_attacks(board, color).values():
        result.update(attacks)

    return result


def control_count(
    board: cshogi.Board,
    color: int,
) -> int:
    """指定陣営が利かせている異なるマス数。"""
    return len(controlled_squares(board, color))


def total_attack_count(
    board: cshogi.Board,
    color: int,
) -> int:
    """
    全駒の利きマス数の合計。

    同じマスを複数の駒が利いていれば、それぞれを数える。
    """
    return sum(
        len(attacks)
        for attacks in all_attacks(board, color).values()
    )


# ---------------------------------------------------------------------------
# F06 駒の利き
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class F06Control:
    black: float
    white: float
    difference: float
    black_unique_squares: int
    white_unique_squares: int
    black_total_attacks: int
    white_total_attacks: int


def f06_control(board: cshogi.Board) -> F06Control:
    """
    F06 駒の利き。

    black / white:
        各陣営が利かせている異なるマス数

    difference:
        Black - White

    black_total_attacks / white_total_attacks:
        駒ごとの利き数を合計した値
    """
    black_unique = control_count(board, BLACK)
    white_unique = control_count(board, WHITE)

    black_total = total_attack_count(board, BLACK)
    white_total = total_attack_count(board, WHITE)

    return F06Control(
        black=float(black_unique),
        white=float(white_unique),
        difference=float(black_unique - white_unique),
        black_unique_squares=black_unique,
        white_unique_squares=white_unique,
        black_total_attacks=black_total,
        white_total_attacks=white_total,
    )


# ---------------------------------------------------------------------------
# F07 駒の移動可能性
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class F07Mobility:
    black_legal_moves: int
    white_legal_moves: int
    difference: int
    black_piece_count: int
    white_piece_count: int
    black_average: float
    white_average: float
    average_difference: float


def _piece_count_on_board(
    board: cshogi.Board,
    color: int,
) -> int:
    count = 0

    for square in range(81):
        piece = board.piece(square)

        if piece != 0 and piece_color(piece) == color:
            count += 1

    return count


def f07_mobility(board: cshogi.Board) -> F07Mobility:
    """
    F07 駒の移動可能性。

    合法手数を陣営ごとに集計する。

    cshogi.Board.legal_moves は現在手番側のみの合法手を返すため、
    盤面をコピーして手番を切り替え、Black / White双方を計算する。
    """
    black_board = board.copy()
    black_board.turn = BLACK

    white_board = board.copy()
    white_board.turn = WHITE

    black_moves = legal_move_count(black_board)
    white_moves = legal_move_count(white_board)

    black_pieces = _piece_count_on_board(board, BLACK)
    white_pieces = _piece_count_on_board(board, WHITE)

    black_average = (
        black_moves / black_pieces
        if black_pieces > 0
        else 0.0
    )

    white_average = (
        white_moves / white_pieces
        if white_pieces > 0
        else 0.0
    )

    return F07Mobility(
        black_legal_moves=black_moves,
        white_legal_moves=white_moves,
        difference=black_moves - white_moves,
        black_piece_count=black_pieces,
        white_piece_count=white_pieces,
        black_average=black_average,
        white_average=white_average,
        average_difference=black_average - white_average,
    )


# ---------------------------------------------------------------------------
# F08 重要地点への利き
# ---------------------------------------------------------------------------

def king_area(
    board: cshogi.Board,
    color: int,
    radius: int = 1,
) -> Set[int]:
    """指定陣営の玉周辺マスを返す。"""
    ksq = king_square(board, color)

    if ksq is None:
        return set()

    kf, kr = square_to_file_rank(ksq)

    result: Set[int] = set()

    for df in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            if df == 0 and dr == 0:
                continue

            file = kf + df
            rank = kr + dr

            square = file_rank_to_square(file, rank)

            if square is not None:
                result.add(square)

    return result


def center_area(radius: int = 1) -> Set[int]:
    """盤中央付近のマスを返す。"""
    result: Set[int] = set()

    center_file = 4
    center_rank = 4

    for df in range(-radius, radius + 1):
        for dr in range(-radius, radius + 1):
            if df == 0 and dr == 0:
                continue

            square = file_rank_to_square(
                center_file + df,
                center_rank + dr,
            )

            if square is not None:
                result.add(square)

    return result


def important_squares(
    board: cshogi.Board,
    color: int,
) -> Set[int]:
    """
    F08で使用する重要地点。

    現段階では、

        1. 盤中央
        2. 相手玉周辺

    を基本とする。

    攻撃拠点候補についてはF14でより明示的に扱う。
    """
    result = set(center_area(radius=1))

    enemy = opponent(color)

    result.update(
        king_area(
            board,
            enemy,
            radius=1,
        )
    )

    return result


def control_on_squares(
    board: cshogi.Board,
    color: int,
    squares: Set[int],
) -> int:
    """指定陣営が重要地点集合に持つ利きの数。"""
    count = 0

    for attacks in all_attacks(board, color).values():
        count += len(attacks.intersection(squares))

    return count


@dataclass(frozen=True)
class F08ImportantControl:
    black: float
    white: float
    difference: float
    black_squares: int
    white_squares: int


def f08_important_control(
    board: cshogi.Board,
) -> F08ImportantControl:
    """
    F08 重要地点への利き。
    """
    black_targets = important_squares(board, BLACK)
    white_targets = important_squares(board, WHITE)

    # 両陣営が比較可能なよう、重要地点を統合する。
    targets = black_targets.union(white_targets)

    black = control_on_squares(
        board,
        BLACK,
        targets,
    )

    white = control_on_squares(
        board,
        WHITE,
        targets,
    )

    return F08ImportantControl(
        black=float(black),
        white=float(white),
        difference=float(black - white),
        black_squares=black,
        white_squares=white,
    )


# ---------------------------------------------------------------------------
# F09 遊び駒の改善
# ---------------------------------------------------------------------------

def piece_activity_score(
    board: cshogi.Board,
    square: int,
) -> float:
    """
    1駒単位の簡易活動性。

    現段階では、

        利き数
        +
        重要地点への利き

    を基礎とする。

    F07の合法手数とは分離する。
    """
    piece = board.piece(square)

    if piece == 0:
        return 0.0

    color = piece_color(piece)

    if color is None:
        return 0.0

    attacks = attacks_from_square(
        board,
        square,
    )

    targets = important_squares(
        board,
        color,
    )

    important = len(attacks.intersection(targets))

    return float(
        len(attacks) + important
    )


def idle_pieces(
    board: cshogi.Board,
    color: int,
    activity_threshold: float = 1.0,
) -> Set[int]:
    """
    活動性が低い駒の集合。

    現段階の定義:
        利き数 + 重要地点への利き <= threshold

    これは研究用の暫定定義であり、実験によって
    thresholdや判定方法を調整する。
    """
    result: Set[int] = set()

    for square in range(81):
        piece = board.piece(square)

        if piece == 0:
            continue

        if piece_color(piece) != color:
            continue

        if piece_activity_score(board, square) <= activity_threshold:
            result.add(square)

    return result


@dataclass(frozen=True)
class F09IdleImprovement:
    black_idle: int
    white_idle: int
    difference: int


def f09_idle_improvement(
    before: cshogi.Board,
    after: cshogi.Board,
) -> F09IdleImprovement:
    """
    F09 遊び駒の改善。

    IdleChange = Idle(S0) - Idle(Sv)

    正の値:
        遊び駒が減った

    負の値:
        遊び駒が増えた
    """
    black_before = len(idle_pieces(before, BLACK))
    black_after = len(idle_pieces(after, BLACK))

    white_before = len(idle_pieces(before, WHITE))
    white_after = len(idle_pieces(after, WHITE))

    black_change = black_before - black_after
    white_change = white_before - white_after

    return F09IdleImprovement(
        black_idle=black_change,
        white_idle=white_change,
        difference=black_change - white_change,
    )


# ---------------------------------------------------------------------------
# F10 活動性
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class F10Activity:
    black: float
    white: float
    difference: float


def _raw_activity(
    board: cshogi.Board,
    color: int,
) -> float:
    """
    F10の活動性。

    現段階の構成:

        Activity =
            a1 * Control
          + a2 * Mobility
          + a3 * ImportantControl

    係数は暫定値。
    学習・実験段階で調整する。
    """
    f06 = f06_control(board)
    f07 = f07_mobility(board)
    f08 = f08_important_control(board)

    if color == BLACK:
        control = f06.black
        mobility = float(f07.black_average)
        important = f08.black
    else:
        control = f06.white
        mobility = float(f07.white_average)
        important = f08.white

    # 暫定係数
    a1 = 1.0
    a2 = 2.0
    a3 = 1.0

    return (
        a1 * control
        + a2 * mobility
        + a3 * important
    )


def f10_activity(
    board: cshogi.Board,
    normalization_c: float = 50.0,
) -> F10Activity:
    """
    F10 活動性。

    生値をtanhで[-1, 1]へ正規化する。
    """
    black_raw = _raw_activity(board, BLACK)
    white_raw = _raw_activity(board, WHITE)

    black = tanh_normalize(
        black_raw,
        normalization_c,
    )

    white = tanh_normalize(
        white_raw,
        normalization_c,
    )

    return F10Activity(
        black=black,
        white=white,
        difference=black - white,
    )


# ---------------------------------------------------------------------------
# Combined extraction
# ---------------------------------------------------------------------------

def extract_activity_features(
    board: cshogi.Board,
) -> Dict[str, object]:
    """F06～F08、F10をまとめて抽出する。"""
    return {
        "F06": f06_control(board),
        "F07": f07_mobility(board),
        "F08": f08_important_control(board),
        "F10": f10_activity(board),
    }
