"""
features/formation.py

F28:
    形成変化

設計:
    F27:
        自玉に対する防御駒の配置

    F28:
        盤面上における駒同士の連携・形成

F28では以下の3要素を利用する。

    1. 駒同士の距離
    2. 味方駒同士の相互支援
    3. 攻撃・防御領域の接続性

F28は形成の「良し悪し」そのものではなく、
駒がどの程度連携した配置になっているかを表す。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict

from .activity import (
    BLACK,
    WHITE,
    all_attacks,
)
from .common import (
    board_pieces,
    square_distance,
    tanh_normalize,
)


# ============================================================
# 重み
# ============================================================

# 同軍駒同士の距離による連携。
RELATIONSHIP_WEIGHT = 1.0

# 味方駒同士の相互支援。
MUTUAL_CONTROL_WEIGHT = 1.0

# 攻撃・防御領域の接続性。
ZONE_CONNECTIVITY_WEIGHT = 1.0

# 正規化スケール。
# 大量棋譜による分布確認後に調整する。
NORMALIZATION_SCALE = 10.0


# ============================================================
# データクラス
# ============================================================

@dataclass(frozen=True)
class F28Formation:
    """
    指定した陣営の形成状態。
    """

    score: float
    relationship: float
    mutual_control: float
    zone_connectivity: float


@dataclass(frozen=True)
class F28FormationChange:
    """
    F28の指し手前後の変化量。
    """

    black: float
    white: float
    difference: float


# ============================================================
# 1. 駒同士の関係
# ============================================================

def piece_relationship_score(
    board,
    color: int,
) -> float:
    """
    同じ陣営の駒同士の距離から形成の連携度を計算する。

    距離が近い駒ほど大きな値を与える。

        relationship = 1 / (1 + distance)

    玉も形成を構成する駒として扱う。

    ただし、同じ陣営の駒が少ない場合は0とする。
    """

    pieces = board_pieces(board, color)

    if len(pieces) < 2:
        return 0.0

    total = 0.0
    pair_count = 0

    for piece_a, piece_b in combinations(pieces, 2):
        distance = square_distance(
            piece_a.square,
            piece_b.square,
        )

        total += 1.0 / (1.0 + distance)
        pair_count += 1

    if pair_count == 0:
        return 0.0

    # 駒数による単純な増加を避けるため平均値にする。
    return total / pair_count


# ============================================================
# 2. 味方駒同士の相互支援
# ============================================================

def mutual_control_score(
    board,
    color: int,
) -> float:
    """
    味方駒が別の味方駒を支えている程度を計算する。

    Aの利きの中にBの駒が存在する場合、

        A -> B

    を1つの支援関係として数える。

    同じ駒同士は除外する。
    """

    pieces = board_pieces(board, color)

    if len(pieces) < 2:
        return 0.0

    attacks = all_attacks(board, color)

    supported_pairs = 0
    possible_pairs = 0

    piece_squares = {
        piece.square
        for piece in pieces
    }

    for piece in pieces:
        targets = attacks.get(piece.square, set())

        # 自分以外の味方駒が存在するマス。
        supported = targets.intersection(piece_squares)
        supported.discard(piece.square)

        supported_pairs += len(supported)

    possible_pairs = len(pieces) * (len(pieces) - 1)

    if possible_pairs == 0:
        return 0.0

    return supported_pairs / possible_pairs


# ============================================================
# 3. 攻撃・防御領域の接続性
# ============================================================

def zone_connectivity_score(
    board,
    color: int,
) -> float:
    """
    自軍駒が作る利き領域の重なりを評価する。

    同じマスを複数の味方駒が制御している場合、
    そのマスを形成の接続点として評価する。

    1つの駒しか利いていないマスより、
    複数駒が利いているマスを高く評価する。
    """

    attacks = all_attacks(board, color)

    if not attacks:
        return 0.0

    control_count: Dict[int, int] = {}

    for targets in attacks.values():
        for target in targets:
            control_count[target] = (
                control_count.get(target, 0) + 1
            )

    if not control_count:
        return 0.0

    total_control = sum(control_count.values())

    if total_control == 0:
        return 0.0

    # 複数の駒から制御されているマスの割合。
    connected_control = sum(
        count
        for count in control_count.values()
        if count >= 2
    )

    return connected_control / total_control


# ============================================================
# F28本体
# ============================================================

def f28_formation(
    board,
    color: int,
) -> F28Formation:
    """
    指定した陣営のF28を計算する。
    """

    relationship = piece_relationship_score(
        board,
        color,
    )

    mutual_control = mutual_control_score(
        board,
        color,
    )

    zone_connectivity = zone_connectivity_score(
        board,
        color,
    )

    raw_score = (
        RELATIONSHIP_WEIGHT * relationship
        + MUTUAL_CONTROL_WEIGHT * mutual_control
        + ZONE_CONNECTIVITY_WEIGHT * zone_connectivity
    )

    score = tanh_normalize(
        raw_score,
        NORMALIZATION_SCALE,
    )

    return F28Formation(
        score=score,
        relationship=relationship,
        mutual_control=mutual_control,
        zone_connectivity=zone_connectivity,
    )


# ============================================================
# 変化量
# ============================================================

def f28_formation_change(
    before,
    after,
) -> F28FormationChange:
    """
    指し手前後のF28変化量を計算する。
    """

    black_before = f28_formation(
        before,
        BLACK,
    )

    black_after = f28_formation(
        after,
        BLACK,
    )

    white_before = f28_formation(
        before,
        WHITE,
    )

    white_after = f28_formation(
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

    return F28FormationChange(
        black=black_change,
        white=white_change,
        difference=black_change - white_change,
    )