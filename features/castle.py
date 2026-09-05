"""
features/castle.py

F29:
    キャッスル進展

特定の囲い名を分類するのではなく、
自玉を中心とした防御形成の進展度を
連続値として表現する。

構成要素:

    1. 玉位置
    2. 金銀配置
    3. 自玉周辺の防御駒密度

各要素は0～1に正規化し、
最終的なF29も0～1の範囲に収める。

F29は囲いの「良し悪し」ではなく、
囲い形成の状態を表す。
"""

from __future__ import annotations

from dataclasses import dataclass

from .activity import (
    BLACK,
    WHITE,
    defense_area,
)
from .common import (
    board_pieces,
    king_square,
    square_distance,
    square_to_file_rank,
)


# ----------------------------------------------------------------------
# 重み
# ----------------------------------------------------------------------

KING_POSITION_WEIGHT = 0.5
GOLD_SILVER_WEIGHT = 1.5
DEFENSE_DENSITY_WEIGHT = 1.5


# ----------------------------------------------------------------------
# 正規化用定数
# ----------------------------------------------------------------------

MAX_KING_POSITION = 4.0
MAX_DEFENSE_DENSITY = 8.0


# ----------------------------------------------------------------------
# データクラス
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class F29CastleProgress:
    """F29のキャッスル進展度."""

    score: float
    king_position: float
    gold_silver: float
    defense_density: float


@dataclass(frozen=True)
class F29CastleProgressChange:
    """F29の局面変化."""

    black: float
    white: float
    difference: float


# ----------------------------------------------------------------------
# ① 玉位置
# ----------------------------------------------------------------------

def king_position_score(
    board,
    color: int,
) -> float:
    """
    玉の横方向への移動度を0～1に正規化する。

    玉が5筋から離れるほど値が大きくなる。

    ただし、玉位置だけでは囲い形成を判断できないため、
    最終スコアへの寄与は小さくする。
    """

    king = king_square(
        board,
        color,
    )

    file_, _ = square_to_file_rank(
        king,
    )

    center_file = 4

    distance = abs(
        file_ - center_file,
    )

    return min(
        float(distance) / MAX_KING_POSITION,
        1.0,
    )


# ----------------------------------------------------------------------
# ② 金銀配置
# ----------------------------------------------------------------------

def gold_silver_placement_score(
    board,
    color: int,
) -> float:
    """
    玉に対する金銀の配置を0～1に正規化する。

    金・銀が玉に近いほど値を大きくする。
    """

    king = king_square(
        board,
        color,
    )

    total = 0.0

    for piece in board_pieces(
        board,
        color,
    ):

        # Gold = 7
        # Silver = 4
        if piece.base_type not in (4, 7):
            continue

        distance = square_distance(
            piece.square,
            king,
        )

        total += 1.0 / (
            1.0 + distance
        )

    # 基準値4で正規化
    return min(
        total / 4.0,
        1.0,
    )


# ----------------------------------------------------------------------
# ③ 防御駒密度
# ----------------------------------------------------------------------

def defense_density_score(
    board,
    color: int,
) -> float:
    """
    自玉周辺の防御駒密度を0～1に正規化する。

    自玉周囲8マスに存在する自駒数を数える。
    """

    area = defense_area(
        board,
        color,
    )

    count = 0

    for piece in board_pieces(
        board,
        color,
    ):

        if piece.square in area:
            count += 1

    return min(
        float(count) / MAX_DEFENSE_DENSITY,
        1.0,
    )


# ----------------------------------------------------------------------
# F29本体
# ----------------------------------------------------------------------

def f29_castle_progress(
    board,
    color: int,
) -> F29CastleProgress:
    """
    F29キャッスル進展度を計算する。

    F29は以下の3要素から構成する。

        ・玉位置
        ・金銀配置
        ・自玉周辺の防御駒密度

    いずれも0～1に正規化する。
    """

    king_position = king_position_score(
        board,
        color,
    )

    gold_silver = gold_silver_placement_score(
        board,
        color,
    )

    defense_density = defense_density_score(
        board,
        color,
    )

    weighted_sum = (
        KING_POSITION_WEIGHT * king_position
        + GOLD_SILVER_WEIGHT * gold_silver
        + DEFENSE_DENSITY_WEIGHT * defense_density
    )

    total_weight = (
        KING_POSITION_WEIGHT
        + GOLD_SILVER_WEIGHT
        + DEFENSE_DENSITY_WEIGHT
    )

    score = (
        weighted_sum
        / total_weight
    )

    score = max(
        0.0,
        min(score, 1.0),
    )

    return F29CastleProgress(
        score=score,
        king_position=king_position,
        gold_silver=gold_silver,
        defense_density=defense_density,
    )


# ----------------------------------------------------------------------
# 局面変化
# ----------------------------------------------------------------------

def f29_castle_progress_change(
    before,
    after,
) -> F29CastleProgressChange:
    """
    局面変化によるF29の変化量を計算する。
    """

    black_before = f29_castle_progress(
        before,
        BLACK,
    )

    black_after = f29_castle_progress(
        after,
        BLACK,
    )

    white_before = f29_castle_progress(
        before,
        WHITE,
    )

    white_after = f29_castle_progress(
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

    return F29CastleProgressChange(
        black=black_change,
        white=white_change,
        difference=(
            black_change
            - white_change
        ),
    )