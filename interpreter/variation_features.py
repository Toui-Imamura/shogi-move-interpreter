"""
interpreter/variation_features.py

1本の探索variationについて特徴量を計算する。

構造:

    S0 -> S1 -> S2 -> ... -> Sv

各遷移:
    ΔFi(t) = Fi(S[t+1]) - Fi(S[t])

variation全体:
    F13
    F15

先頭の遷移:
    F34
    F35
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cshogi

from features.attack import (
    f13_material_gain_process,
    f15_attack_continuity,
)

from features.consistency import compute_consistency

from features.strategy import compute_strategy_direction

from features.persistence import compute_persistence

from features.transition import (
    make_usi_variation,
    position_at,
)

from interpreter.feature_adapters import (
    compute_f01_f33_deltas,
)


@dataclass(frozen=True)
class VariationFeatureResult:
    """
    1本のvariationに対する特徴量計算結果。
    """

    positions: Sequence[cshogi.Board]

    transition_deltas: Sequence[dict[str, float]]

    variation_deltas: dict[str, float]


def compute_variation_features(
    initial_board: cshogi.Board,
    usi_moves: Sequence[str],
) -> VariationFeatureResult:
    """
    1本のvariationについて特徴量を計算する。

    Parameters
    ----------
    initial_board:
        variation開始時の局面。

    usi_moves:
        USI形式の指し手列。

    Returns
    -------
    VariationFeatureResult
        positions:
            S0〜Sv

        transition_deltas:
            各遷移におけるF01〜F33の変化

        variation_deltas:
            variation全体または先頭手から求める
            F13 / F15 / F34 / F35
    """

    if initial_board is None:
        raise ValueError(
            "initial_board must not be None"
        )

    if usi_moves is None:
        raise ValueError(
            "usi_moves must not be None"
        )

    moves = list(usi_moves)

    variation = make_usi_variation(
        initial_board,
        moves,
    )

    positions = [
        position_at(variation, index)
        for index in range(len(moves) + 1)
    ]

    # --------------------------------------------------
    # 各遷移のF01〜F33
    # --------------------------------------------------

    transition_deltas: list[dict[str, float]] = []

    for index in range(len(positions) - 1):
        before = positions[index]
        after = positions[index + 1]

        # cshogi 1.0.4で確認済みのAPI
        move = before.move_from_usi(
            moves[index]
        )

        deltas = compute_f01_f33_deltas(
            before,
            after,
            move=move,
        )

        transition_deltas.append(deltas)

    # --------------------------------------------------
    # variation全体の特徴量
    # --------------------------------------------------

    variation_deltas: dict[str, float] = {}

    if moves:
        # --------------------------------------------------
        # F13: Material Gain Process
        # F15: Attack Continuity
        # --------------------------------------------------

        f13 = f13_material_gain_process(
            variation
        )

        f15 = f15_attack_continuity(
            variation
        )

        variation_deltas["F13"] = float(
            f13.difference
        )

        variation_deltas["F15"] = float(
            f15.difference
        )

        # --------------------------------------------------
        # F34: Multi-feature Strategy Direction
        # --------------------------------------------------

        first_transition = transition_deltas[0]

        f34 = compute_strategy_direction(
            first_transition
        )

        variation_deltas["F34_material"] = float(
            f34.material
        )

        variation_deltas["F34_attack"] = float(
            f34.attack
        )

        variation_deltas["F34_defense"] = float(
            f34.defense
        )

        variation_deltas["F34_activity"] = float(
            f34.activity
        )

        variation_deltas["F34_formation"] = float(
            f34.formation
        )

        # --------------------------------------------------
        # F35: Multi-feature Consistency
        # --------------------------------------------------

        f35 = compute_consistency(
            first_transition
        )

        variation_deltas["F35"] = float(
            f35.score
        )

        # --------------------------------------------------
        # F36: Change Persistence
        # --------------------------------------------------

        feature_time_series: dict[str, list[float]] = {}

        for deltas in transition_deltas:
            for feature_name, delta in deltas.items():
                feature_time_series.setdefault(
                    feature_name,
                    []
                ).append(float(delta))

        f36 = compute_persistence(
            feature_time_series
        )

        variation_deltas["F36"] = float(
            f36.score
        )

    return VariationFeatureResult(
        positions=positions,
        transition_deltas=transition_deltas,
        variation_deltas=variation_deltas,
    )
