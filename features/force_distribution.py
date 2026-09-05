"""
F33: 戦力分布変化

盤面全体における自軍戦力の分布を評価する。

F33では、盤面を空間的な領域に分け、
各領域にどれだけの戦力が存在するかを表現する。

戦力は駒の価値を基本として計算する。

注意:
    F21は攻撃の集中度を評価する特徴量であるのに対し、
    F33は盤面全体における「戦力そのものの空間分布」を評価する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cshogi

from .common import (
    BLACK,
    WHITE,
    board_pieces,
    tanh_normalize,
)


# ---------------------------------------------------------
# 定数
# ---------------------------------------------------------

EPSILON = 1e-8

# 分布領域
REGIONS = (
    "left",
    "center",
    "right",
    "own_side",
    "enemy_side",
)


# ---------------------------------------------------------
# データクラス
# ---------------------------------------------------------

@dataclass(frozen=True)
class F33ForceDistribution:
    """
    F33の戦力分布。

    score:
        戦力分布を表す代表値。

    left / center / right:
        左・中央・右方向への戦力分布。

    own_side / enemy_side:
        自陣側・敵陣側への戦力分布。

    vector:
        分布ベクトル。
    """

    score: float
    left: float
    center: float
    right: float
    own_side: float
    enemy_side: float

    @property
    def vector(self) -> Tuple[float, ...]:
        return (
            self.left,
            self.center,
            self.right,
            self.own_side,
            self.enemy_side,
        )


@dataclass(frozen=True)
class F33ForceDistributionChange:
    """
    F33の戦力分布変化。

    black:
        黒側の変化。

    white:
        白側の変化。

    difference:
        黒側変化 - 白側変化。
    """

    black: F33ForceDistribution
    white: F33ForceDistribution
    difference: Tuple[float, ...]


# ---------------------------------------------------------
# 基本処理
# ---------------------------------------------------------

def _usi_position(square: int) -> Tuple[int, int]:
    """
    cshogiのUSI座標を数値座標に変換する。

    Returns
    -------
    file:
        1～9の筋。

    rank:
        1～9の段。
    """

    usi = cshogi.SQUARE_NAMES[square]

    file_ = int(usi[0])
    rank_char = usi[1]

    rank = ord(rank_char) - ord("a") + 1

    return file_, rank


def _piece_region(
    square: int,
    color: int,
) -> Tuple[str, ...]:
    """
    駒が属する空間領域を返す。

    1つの駒について、
    横方向の領域(left / center / right)と
    縦方向の領域(own_side / enemy_side)を返す。
    """

    file_, rank = _usi_position(square)

    regions = []

    # -----------------------------------------------------
    # 横方向
    #
    # 1～3筋 : left
    # 4～6筋 : center
    # 7～9筋 : right
    # -----------------------------------------------------

    if file_ <= 3:
        regions.append("left")
    elif file_ <= 6:
        regions.append("center")
    else:
        regions.append("right")

    # -----------------------------------------------------
    # 縦方向
    #
    # 黒:
    #   1～4段が敵陣側
    #   5～9段が自陣側
    #
    # 白:
    #   6～9段が敵陣側
    #   1～5段が自陣側
    #
    # ただし、ここでは厳密な「敵陣」ではなく、
    # 盤面上の自陣側/敵陣側という相対的な方向を表す。
    # -----------------------------------------------------

    if color == BLACK:
        if rank <= 4:
            regions.append("enemy_side")
        else:
            regions.append("own_side")

    elif color == WHITE:
        if rank >= 6:
            regions.append("enemy_side")
        else:
            regions.append("own_side")

    else:
        raise ValueError(f"invalid color: {color}")

    return tuple(regions)


def _region_raw_values(
    board: cshogi.Board,
    color: int,
) -> Dict[str, float]:
    """
    各領域に存在する自軍戦力を計算する。
    """

    if color not in (BLACK, WHITE):
        raise ValueError(f"invalid color: {color}")

    values = {region: 0.0 for region in REGIONS}

    for piece in board_pieces(board):
        if piece.color != color:
            continue

        for region in _piece_region(piece.square, color):
            values[region] += piece.value

    return values


def _normalize_distribution(
    values: Dict[str, float],
) -> Dict[str, float]:
    """
    戦力分布を正規化する。

    横方向:
        left + center + right = 1

    縦方向:
        own_side + enemy_side = 1

    それぞれを独立して正規化する。
    """

    horizontal_total = (
        values["left"]
        + values["center"]
        + values["right"]
    )

    vertical_total = (
        values["own_side"]
        + values["enemy_side"]
    )

    normalized = {
        "left": 0.0,
        "center": 0.0,
        "right": 0.0,
        "own_side": 0.0,
        "enemy_side": 0.0,
    }

    if horizontal_total > EPSILON:
        normalized["left"] = (
            values["left"] / horizontal_total
        )
        normalized["center"] = (
            values["center"] / horizontal_total
        )
        normalized["right"] = (
            values["right"] / horizontal_total
        )

    if vertical_total > EPSILON:
        normalized["own_side"] = (
            values["own_side"] / vertical_total
        )
        normalized["enemy_side"] = (
            values["enemy_side"] / vertical_total
        )

    return normalized


# ---------------------------------------------------------
# F33
# ---------------------------------------------------------

def f33_force_distribution(
    board: cshogi.Board,
    color: int,
) -> F33ForceDistribution:
    """
    F33 戦力分布を計算する。
    """

    raw_values = _region_raw_values(board, color)
    distribution = _normalize_distribution(raw_values)

    # 代表値:
    # 左右の偏り・自陣/敵陣への偏りをまとめる。
    #
    # 中央への集中を基準とし、
    # 敵陣側への進出をプラス、
    # 自陣側への偏りをマイナスとして表現する。
    #
    # ただしこれは「良さ」ではなく、
    # あくまで戦力配置の方向を表す値である。

    score_raw = (
        distribution["enemy_side"]
        - distribution["own_side"]
    )

    score = tanh_normalize(
        score_raw,
        scale=1.0,
    )

    return F33ForceDistribution(
        score=score,
        left=distribution["left"],
        center=distribution["center"],
        right=distribution["right"],
        own_side=distribution["own_side"],
        enemy_side=distribution["enemy_side"],
    )


def f33_force_distribution_change(
    before: cshogi.Board,
    after: cshogi.Board,
    color: int,
) -> F33ForceDistribution:
    """
    F33のbefore -> afterの変化を計算する。
    """

    before_feature = f33_force_distribution(
        before,
        color,
    )

    after_feature = f33_force_distribution(
        after,
        color,
    )

    return F33ForceDistribution(
        score=after_feature.score - before_feature.score,
        left=after_feature.left - before_feature.left,
        center=after_feature.center - before_feature.center,
        right=after_feature.right - before_feature.right,
        own_side=after_feature.own_side - before_feature.own_side,
        enemy_side=after_feature.enemy_side - before_feature.enemy_side,
    )


def f33_force_distribution_change_both(
    before: cshogi.Board,
    after: cshogi.Board,
) -> F33ForceDistributionChange:
    """
    黒・白双方のF33変化を計算する。
    """

    black = f33_force_distribution_change(
        before,
        after,
        BLACK,
    )

    white = f33_force_distribution_change(
        before,
        after,
        WHITE,
    )

    difference = tuple(
        b - w
        for b, w in zip(
            black.vector,
            white.vector,
        )
    )

    return F33ForceDistributionChange(
        black=black,
        white=white,
        difference=difference,
    )


# ---------------------------------------------------------
# 互換用エイリアス
# ---------------------------------------------------------

compute_force_distribution = f33_force_distribution

compute_force_distribution_change = (
    f33_force_distribution_change
)