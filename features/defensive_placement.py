"""
defensive_placement.py

F27:
    防御駒配置変化

設計:
    F17:
        玉周辺に存在する守備駒の「数」

    F18:
        玉周辺における自軍・敵軍の「利き」

    F26:
        玉そのものの安全度

    F27:
        玉に対して守備駒が「どのように配置されているか」

F27では単純な守備駒数ではなく、

    1. 玉からの距離
    2. 金・銀の配置
    3. 玉周辺8方向への配置

を利用して、防御駒の配置状態を表現する。

注意:
    F27は指し手の意図そのものではなく、
    局面における防御駒の配置状態を表す特徴量である。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict

import cshogi

from .activity import BLACK, WHITE, defense_area
from .common import (
    iter_pieces,
    king_square,
    square_to_file_rank,
    tanh_normalize,
)


# ------------------------------------------------------------
# 重み
# ------------------------------------------------------------

# 玉への近さ。
# 近い駒ほど防御配置への寄与を大きくする。
DISTANCE_WEIGHT = 1.0

# 金・銀は守備に利用されることが多いため、
# 配置状態への寄与を大きくする。
GOLD_SILVER_WEIGHT = 1.5

# 玉周辺8方向の配置バランス。
DIRECTION_WEIGHT = 1.0

# 最終的な正規化スケール。
# 暫定値。大量棋譜による分布確認後に調整する。
NORMALIZATION_SCALE = 20.0


# ------------------------------------------------------------
# データクラス
# ------------------------------------------------------------

@dataclass(frozen=True)
class F27DefensivePlacement:
    """
    F27の防御駒配置状態。

    score:
        防御駒配置の総合スコア

    proximity:
        玉への近さを考慮した防御駒配置

    gold_silver:
        金銀の配置

    direction_balance:
        玉周辺8方向への配置状態
    """

    score: float
    proximity: float
    gold_silver: float
    direction_balance: float


@dataclass(frozen=True)
class F27DefensivePlacementChange:
    """
    F27の変化量。
    """

    black: float
    white: float
    difference: float


# ------------------------------------------------------------
# 基本処理
# ------------------------------------------------------------

def _distance(board, square_a: int, square_b: int) -> float:
    """
    2つのマスの距離を計算する。

    将棋盤上のファイル・ランクを使ったユークリッド距離。
    """

    file_a, rank_a = square_to_file_rank(square_a)
    file_b, rank_b = square_to_file_rank(square_b)

    return sqrt(
        (file_a - file_b) ** 2
        + (rank_a - rank_b) ** 2
    )


def _normalized_proximity(distance: float) -> float:
    """
    玉からの距離を0～1程度の範囲に変換する。

    距離1:
        1.0

    距離2:
        0.5

    距離3:
        0.333...

    というように、近い駒を強く評価する。
    """

    return 1.0 / (1.0 + distance)


# ------------------------------------------------------------
# 1. 玉への近さ
# ------------------------------------------------------------

def defensive_proximity_score(board, color: int) -> float:
    """
    自玉周辺の防御駒について、
    玉からの距離を考慮した配置スコアを計算する。

    F17とは異なり、
    単純な駒数ではなく玉との距離を利用する。
    """

    king = king_square(board, color)

    if king is None:
        return 0.0

    score = 0.0

    for piece in iter_pieces(board):
        if piece.color != color:
            continue

        distance = _distance(board, king, piece.square)

        # 玉から遠すぎる駒は防御配置への寄与を小さくする。
        proximity = _normalized_proximity(distance)

        score += proximity

    return score


# ------------------------------------------------------------
# 2. 金銀配置
# ------------------------------------------------------------

def gold_silver_placement_score(board, color: int) -> float:
    """
    玉に対する金・銀の配置スコア。

    金・銀が玉に近いほど高く評価する。

    これは単なる金銀の枚数ではなく、
    「玉からどこに配置されているか」を評価する。
    """

    king = king_square(board, color)

    if king is None:
        return 0.0

    score = 0.0

    for piece in iter_pieces(board):
        if piece.color != color:
            continue

        if piece.base_type not in (cshogi.GOLD, cshogi.SILVER):
            continue

        distance = _distance(board, king, piece.square)
        proximity = _normalized_proximity(distance)

        score += proximity

    return score


# ------------------------------------------------------------
# 3. 玉周辺8方向
# ------------------------------------------------------------

def direction_distribution(
    board,
    color: int,
) -> Dict[str, int]:
    """
    玉周辺8方向に存在する自軍駒の数を返す。

    direction:
        forward
        backward
        left
        right
        forward_left
        forward_right
        backward_left
        backward_right

    forwardは各陣営の「前方向」とする。

    Black:
        rankが小さい方向

    White:
        rankが大きい方向
    """

    king = king_square(board, color)

    directions = {
        "forward": 0,
        "backward": 0,
        "left": 0,
        "right": 0,
        "forward_left": 0,
        "forward_right": 0,
        "backward_left": 0,
        "backward_right": 0,
    }

    if king is None:
        return directions

    king_file, king_rank = square_to_file_rank(king)

    for piece in iter_pieces(board):
        if piece.color != color:
            continue

        file_, rank = square_to_file_rank(piece.square)

        df = file_ - king_file
        dr = rank - king_rank

        # 玉自身は除外
        if df == 0 and dr == 0:
            continue

        # 玉周辺1マスのみを対象にする
        if abs(df) > 1 or abs(dr) > 1:
            continue

        # Blackはrank減少方向、Whiteはrank増加方向を前とする
        if color == BLACK:
            forward_dr = -dr
        else:
            forward_dr = dr

        if forward_dr > 0 and df == 0:
            directions["forward"] += 1
        elif forward_dr < 0 and df == 0:
            directions["backward"] += 1
        elif df < 0 and forward_dr == 0:
            directions["left"] += 1
        elif df > 0 and forward_dr == 0:
            directions["right"] += 1
        elif df < 0 and forward_dr > 0:
            directions["forward_left"] += 1
        elif df > 0 and forward_dr > 0:
            directions["forward_right"] += 1
        elif df < 0 and forward_dr < 0:
            directions["backward_left"] += 1
        elif df > 0 and forward_dr < 0:
            directions["backward_right"] += 1

    return directions


def direction_balance_score(board, color: int) -> float:
    """
    玉周辺8方向の防御配置を評価する。

    防御駒が複数方向に配置されているほど、
    一方向への偏りが小さくなる。

    現段階では、
        使用されている方向数 / 8
    を基本スコアとする。
    """

    distribution = direction_distribution(board, color)

    active_directions = sum(
        1
        for count in distribution.values()
        if count > 0
    )

    return active_directions / 8.0


# ------------------------------------------------------------
# F27本体
# ------------------------------------------------------------

def f27_defensive_placement(
    board,
    color: int,
) -> F27DefensivePlacement:
    """
    指定した陣営のF27を計算する。
    """

    proximity = defensive_proximity_score(board, color)

    gold_silver = gold_silver_placement_score(board, color)

    direction_balance = direction_balance_score(board, color)

    raw_score = (
        DISTANCE_WEIGHT * proximity
        + GOLD_SILVER_WEIGHT * gold_silver
        + DIRECTION_WEIGHT * direction_balance
    )

    score = tanh_normalize(
        raw_score,
        scale=NORMALIZATION_SCALE,
    )

    return F27DefensivePlacement(
        score=score,
        proximity=proximity,
        gold_silver=gold_silver,
        direction_balance=direction_balance,
    )


def f27_defensive_placement_change(
    before,
    after,
) -> F27DefensivePlacementChange:
    """
    指し手前後のF27変化量を計算する。
    """

    black_before = f27_defensive_placement(
        before,
        BLACK,
    )

    black_after = f27_defensive_placement(
        after,
        BLACK,
    )

    white_before = f27_defensive_placement(
        before,
        WHITE,
    )

    white_after = f27_defensive_placement(
        after,
        WHITE,
    )

    black_change = (
        black_after.score
        - black_before.score
    )

    white_change = (
        white_after.score
        - white_before.score
    )

    return F27DefensivePlacementChange(
        black=black_change,
        white=white_change,
        difference=black_change - white_change,
    )
