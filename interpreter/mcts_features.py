"""
interpreter/mcts_features.py

複数のMCTS探索variationから、
F37: MCTS変化頻度
F38: MCTS訪問重み付き変化
を計算する。

各variationは、

    S0 -> S1 -> S2 -> ... -> Sv

という探索系列を表す。

F37/F38では、原則として
S0 -> Sv
の変化量を使用する。

注意:
    MCTS訪問回数は「指し手の意図の確率」ではない。
    あくまで探索結果を集約するための重みとして扱う。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cshogi

from features.mcts_frequency import (
    F37MCTSChangeFrequency,
    compute_mcts_change_frequency,
)

from features.mcts_weighted import (
    F38VisitWeightedChange,
    MCTSVariation,
    compute_visit_weighted_change,
)

from interpreter.feature_adapters import (
    compute_f01_f33_deltas,
)


@dataclass(frozen=True)
class MCTSFeatureResult:
    """
    複数のMCTS variationに対するF37/F38の計算結果。
    """

    variation_deltas: Sequence[Mapping[str, float]]

    f37: F37MCTSChangeFrequency

    f38: F38VisitWeightedChange


def compute_variation_endpoint_deltas(
    initial_board: cshogi.Board,
    variation_result,
) -> dict[str, float]:
    """
    1本のvariationについて、
    S0 -> Sv の特徴量変化を計算する。

    F37/F38では各探索系列の最終局面までに
    どの特徴量が変化したかを扱うため、
    途中の1手だけではなく最終局面を使用する。

    Parameters
    ----------
    initial_board:
        variation開始時の局面。

    variation_result:
        interpreter.variation_features.compute_variation_features()
        の戻り値。

    Returns
    -------
    dict[str, float]
        特徴量名 -> S0からSvまでの変化量。
    """

    if initial_board is None:
        raise ValueError(
            "initial_board must not be None"
        )

    if variation_result is None:
        raise ValueError(
            "variation_result must not be None"
        )

    positions = variation_result.positions

    if not positions:
        return {}

    initial = positions[0]
    final = positions[-1]

    # F01〜F33についてS0→Svを計算する。
    #
    # moveは渡さない。
    # ここでは「特定の1手の交換」を評価するのではなく、
    # variation全体の始点と終点を比較するためである。
    endpoint_deltas = compute_f01_f33_deltas(
        initial,
        final,
    )

    result = {
        feature_name: float(delta)
        for feature_name, delta in endpoint_deltas.items()
    }

    # variation全体から直接計算できる特徴量を追加する。
    #
    # F34〜F36は「変化量」そのものではなく、
    # 戦略方向・整合性・持続性を表すため、
    # F37/F38のΔFiとしてはここでは扱わない。
    for feature_name in ("F13", "F15"):
        if feature_name in variation_result.variation_deltas:
            result[feature_name] = float(
                variation_result.variation_deltas[feature_name]
            )

    return result


def compute_mcts_features(
    initial_board: cshogi.Board,
    variations,
) -> MCTSFeatureResult:
    """
    複数のMCTS variationからF37/F38を計算する。

    Parameters
    ----------
    initial_board:
        MCTS開始時の局面。

    variations:
        以下の形式を想定する。

        [
            {
                "result": VariationFeatureResult,
                "visits": 10,
            },
            {
                "result": VariationFeatureResult,
                "visits": 5,
            },
        ]

    Returns
    -------
    MCTSFeatureResult
        variationごとの特徴量変化、
        F37、
        F38。
    """

    if initial_board is None:
        raise ValueError(
            "initial_board must not be None"
        )

    if variations is None:
        raise ValueError(
            "variations must not be None"
        )

    variation_deltas = []
    weighted_variations = []

    for variation in variations:
        if "result" not in variation:
            raise ValueError(
                "each variation must contain 'result'"
            )

        if "visits" not in variation:
            raise ValueError(
                "each variation must contain 'visits'"
            )

        visits = variation["visits"]

        if visits < 0:
            raise ValueError(
                "visits must be non-negative"
            )

        result = variation["result"]

        deltas = compute_variation_endpoint_deltas(
            initial_board,
            result,
        )

        variation_deltas.append(deltas)

        weighted_variations.append(
            MCTSVariation(
                visits=int(visits),
                deltas=deltas,
            )
        )

    # F37
    f37 = compute_mcts_change_frequency(
        variation_deltas
    )

    # F38
    f38 = compute_visit_weighted_change(
        weighted_variations
    )

    return MCTSFeatureResult(
        variation_deltas=variation_deltas,
        f37=f37,
        f38=f38,
    )
