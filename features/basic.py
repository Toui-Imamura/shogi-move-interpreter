"""
features/basic.py

F01～F05 基本特徴量

F01 駒得差
F02 持ち駒価値差
F03 駒交換による戦力変化
F04 大駒バランス
F05 小駒構成
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cshogi

from .common import (
    BLACK,
    WHITE,
    MAJOR_PIECES,
    board_pieces,
    hand_value,
    piece_value,
)


# ============================================================
# F01 駒得差
# ============================================================

@dataclass(frozen=True)
class F01Material:
    black: float
    white: float
    difference: float


def f01_material(board: cshogi.Board) -> F01Material:
    """
    F01 駒得差。

    盤上の駒の価値をBlack/Whiteそれぞれについて計算する。
    持ち駒はF02で別に扱う。
    """

    black = sum(
        piece.value
        for piece in board_pieces(board, BLACK)
    )

    white = sum(
        piece.value
        for piece in board_pieces(board, WHITE)
    )

    return F01Material(
        black=black,
        white=white,
        difference=black - white,
    )


# ============================================================
# F02 持ち駒価値差
# ============================================================

@dataclass(frozen=True)
class F02HandMaterial:
    black: float
    white: float
    difference: float


def f02_hand_material(board: cshogi.Board) -> F02HandMaterial:
    """
    F02 持ち駒価値差。
    """

    black = hand_value(board, BLACK)
    white = hand_value(board, WHITE)

    return F02HandMaterial(
        black=black,
        white=white,
        difference=black - white,
    )


# ============================================================
# F03 駒交換による戦力変化
# ============================================================

@dataclass(frozen=True)
class F03Exchange:
    """
    F03 駒交換による戦力変化。

    1手の指し手について、

        E1 交換発生
        E2 捕獲駒価値
        E3 大駒交換量
        E4 小駒交換量
        E5 盤上→持ち駒
        E6 駒構成変化

    を保持する。

    scoreは現時点ではE1～E6から計算する
    基本スコアであり、最終的なβ係数は
    学習時に変更可能とする。
    """

    exchange_occurred: int

    captured_piece: int
    captured_piece_type: Optional[int]
    captured_piece_value: float

    major_exchange_value: float
    minor_exchange_value: float

    board_to_hand_black: float
    board_to_hand_white: float

    composition_change: float

    score: float


def _is_major_piece(piece_type: Optional[int]) -> bool:
    """
    大駒かどうか。

    大駒:
        角
        飛
        馬
        龍
    """

    if piece_type is None:
        return False

    return piece_type in MAJOR_PIECES


def _is_minor_piece(piece_type: Optional[int]) -> bool:
    """
    小駒として扱える駒か判定する。

    現段階では大駒・玉以外を対象とする。
    """

    if piece_type is None:
        return False

    if piece_type == cshogi.KING:
        return False

    return not _is_major_piece(piece_type)


def _captured_piece_type(
    board: cshogi.Board,
    move: int,
) -> Optional[int]:
    """
    指し手によって捕獲された駒の駒種を取得する。

    cshogi.move_cap(move) は捕獲された駒の
    piece値を返すため、その値からpiece_typeを取得する。
    """

    captured_piece = cshogi.move_cap(move)

    if captured_piece == cshogi.NONE:
        return None

    # move_cap()で取得した駒のpiece値から駒種を取得する。
    #
    # cshogiのpiece_type()は盤上のマスに対して使うため、
    # 捕獲駒についてはpiece値から色を除いた値を求める。
    if captured_piece >= cshogi.WPAWN:
        return captured_piece - cshogi.WPAWN + cshogi.PAWN

    return captured_piece


def f03_exchange(
    before: cshogi.Board,
    after: cshogi.Board,
    move: int,
) -> F03Exchange:
    """
    F03 駒交換による戦力変化。

    before:
        指し手を指す前の局面 S0

    after:
        指し手を指した後の局面 S1

    move:
        S0 -> S1 の指し手

    cshogi 1.0.4のmove APIを利用する。
    """

    if not before.is_legal(move):
        raise ValueError(
            f"illegal move: {move}"
        )

    captured_piece = cshogi.move_cap(move)

    exchange_occurred = (
        1 if captured_piece != cshogi.NONE else 0
    )

    captured_type = _captured_piece_type(
        before,
        move,
    )

    captured_value = (
        piece_value(captured_type)
        if captured_type is not None
        else 0.0
    )

    # --------------------------------------------------------
    # E3 / E4
    # --------------------------------------------------------

    major_exchange_value = 0.0
    minor_exchange_value = 0.0

    if _is_major_piece(captured_type):
        major_exchange_value = captured_value

    elif _is_minor_piece(captured_type):
        minor_exchange_value = captured_value

    # --------------------------------------------------------
    # E5
    #
    # 捕獲した駒は、相手の盤上から
    # 自分の持ち駒へ移動する。
    #
    # before / afterの持ち駒価値差から確認する。
    # --------------------------------------------------------

    before_black_hand = hand_value(
        before,
        BLACK,
    )

    before_white_hand = hand_value(
        before,
        WHITE,
    )

    after_black_hand = hand_value(
        after,
        BLACK,
    )

    after_white_hand = hand_value(
        after,
        WHITE,
    )

    black_hand_delta = (
        after_black_hand - before_black_hand
    )

    white_hand_delta = (
        after_white_hand - before_white_hand
    )

    board_to_hand_black = max(
        0.0,
        black_hand_delta,
    )

    board_to_hand_white = max(
        0.0,
        white_hand_delta,
    )

    # --------------------------------------------------------
    # E6
    #
    # 現段階では捕獲された駒の価値を
    # 構成変化量として扱う。
    #
    # 将来的には、
    #   歩・香・桂・銀・金・角・飛
    # のベクトル変化に拡張する。
    # --------------------------------------------------------

    composition_change = (
        captured_value
        if exchange_occurred
        else 0.0
    )

    # --------------------------------------------------------
    # 基本スコア
    #
    # β係数は後の学習で変更可能。
    # 現段階では各要素をそのまま利用する。
    # --------------------------------------------------------

    score = (
        major_exchange_value
        + minor_exchange_value
        + board_to_hand_black
        + board_to_hand_white
    )

    return F03Exchange(
        exchange_occurred=exchange_occurred,
        captured_piece=captured_piece,
        captured_piece_type=captured_type,
        captured_piece_value=captured_value,
        major_exchange_value=major_exchange_value,
        minor_exchange_value=minor_exchange_value,
        board_to_hand_black=board_to_hand_black,
        board_to_hand_white=board_to_hand_white,
        composition_change=composition_change,
        score=score,
    )


# ============================================================
# F04 大駒バランス
# ============================================================

@dataclass(frozen=True)
class F04MajorBalance:
    black: float
    white: float
    difference: float

    black_count: int
    white_count: int


def f04_major_balance(
    board: cshogi.Board,
) -> F04MajorBalance:
    """
    F04 大駒バランス。

    対象:
        角
        飛
        馬
        龍
    """

    black_pieces = [
        piece
        for piece in board_pieces(board, BLACK)
        if piece.piece_type in MAJOR_PIECES
    ]

    white_pieces = [
        piece
        for piece in board_pieces(board, WHITE)
        if piece.piece_type in MAJOR_PIECES
    ]

    black = sum(
        piece.value
        for piece in black_pieces
    )

    white = sum(
        piece.value
        for piece in white_pieces
    )

    return F04MajorBalance(
        black=black,
        white=white,
        difference=black - white,
        black_count=len(black_pieces),
        white_count=len(white_pieces),
    )


# ============================================================
# F05 小駒構成
# ============================================================

@dataclass(frozen=True)
class F05MinorComposition:
    """
    F05 小駒構成。

    対象:
        歩
        香
        桂
        銀
        金

    成歩・成香・成桂・成銀は
    元の駒種に戻して集計する。
    """

    black: Dict[int, int]
    white: Dict[int, int]
    difference: Dict[int, int]


def f05_minor_composition(
    board: cshogi.Board,
) -> F05MinorComposition:

    target_types = (
        cshogi.PAWN,
        cshogi.LANCE,
        cshogi.KNIGHT,
        cshogi.SILVER,
        cshogi.GOLD,
    )

    black: Dict[int, int] = {
        piece_type: 0
        for piece_type in target_types
    }

    white: Dict[int, int] = {
        piece_type: 0
        for piece_type in target_types
    }

    for piece in board_pieces(board, BLACK):
        if piece.base_type in target_types:
            black[piece.base_type] += 1

    for piece in board_pieces(board, WHITE):
        if piece.base_type in target_types:
            white[piece.base_type] += 1

    difference = {
        piece_type: black[piece_type] - white[piece_type]
        for piece_type in target_types
    }

    return F05MinorComposition(
        black=black,
        white=white,
        difference=difference,
    )


# ============================================================
# F01～F05まとめ
# ============================================================

def extract_basic_features(
    board: cshogi.Board,
) -> Dict[str, object]:
    """
    F01～F05をまとめて取得する。
    """

    return {
        "F01": f01_material(board),
        "F02": f02_hand_material(board),
        "F04": f04_major_balance(board),
        "F05": f05_minor_composition(board),
    }
