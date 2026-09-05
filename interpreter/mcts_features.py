"""
interpreter/mcts_features.py

複数のMCTS探索variationから、

F37: MCTS変化頻度
F38: MCTS訪問重み付き変化
F39: 特徴量変化の集中度

を計算する。

各variation:

    S0 -> S1 -> S2 -> ... -> Sv

F37/F38では、各variationについて
S0 -> Sv の特徴量変化を使用する。

F39では、F38で得られた
訪問重み付き特徴量変化を用いて、
MCTS全体として変化がどの特徴量に集中しているかを計算する。

注意:
    MCTS訪問回数は「指し手の意図の確率」ではない。
    探索結果を集約するための重みとして扱う。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cshogi

from features.change_concentration import (
    compute_feature_change_concentration,
)

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
    複数のMCTS variationに対する
    F37/F38/F39の計算結果。
    """

    variation_deltas: Sequence[Mapping[str, float]]

    f37: F37MCTSChangeFrequency

    f38: F38VisitWeightedChange

    f39: float


def compute_variation_endpoint_deltas(
    initial_board: cshogi.Board,
    variation_result,
) -> dict[str, float]:
    """
    1本のvariationについて、
    S0 -> Sv の特徴量変化を計算する。

    Parameters
    ----------
    initial_board:
        variation開始時の局面。

    variation_result:
        compute_variation_features() の戻り値。

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

    # --------------------------------------------------
    # F01〜F33
    # --------------------------------------------------

    endpoint_deltas = compute_f01_f33_deltas(
        initial,
        final,
    )

    result = {
        feature_name: float(delta)
        for feature_name, delta in endpoint_deltas.items()
    }

    return result


def compute_mcts_features(
    initial_board: cshogi.Board,
    variations,
) -> MCTSFeatureResult:
    """
    複数のMCTS variationから
    F37/F38/F39を計算する。

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
        F38、
        F39。
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

    # --------------------------------------------------
    # F37
    # --------------------------------------------------

    f37 = compute_mcts_change_frequency(
        variation_deltas
    )

    # --------------------------------------------------
    # F38
    # --------------------------------------------------

    f38 = compute_visit_weighted_change(
        weighted_variations
    )

    # --------------------------------------------------
    # F39
    #
    # F38で得られた訪問重み付き変化を、
    # 特徴量変化ベクトルとして扱う。
    # --------------------------------------------------

    f39 = compute_feature_change_concentration(
        f38.weighted_changes
    )

    return MCTSFeatureResult(
        variation_deltas=variation_deltas,
        f37=f37,
        f38=f38,
        f39=float(f39),
    )
