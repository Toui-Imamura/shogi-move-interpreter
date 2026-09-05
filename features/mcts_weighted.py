"""
F38: MCTS訪問重み付き変化

MCTS探索系列ごとの特徴量変化を、
各系列の訪問回数を重みとして加重平均する。

                    Σ N_v * Δf_i^(v)
overline{Δf_i} = ─────────────────────
                       Σ N_v

注意:
    訪問回数は意図の確率ではない。
    MCTS探索結果を集約するための重みとして扱う。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence


EPSILON = 1e-8


@dataclass(frozen=True)
class MCTSVariation:
    """
    1本のMCTS探索系列。

    visits:
        MCTSでその系列に割り当てられた訪問回数。

    deltas:
        特徴量名 -> その探索系列における特徴量変化。
    """

    visits: int
    deltas: Mapping[str, float]


@dataclass(frozen=True)
class F38VisitWeightedChange:
    """
    F38 MCTS訪問重み付き変化。

    weighted_changes:
        特徴量名 -> 訪問回数による加重平均変化量。
    """

    weighted_changes: Dict[str, float]

    def get(
        self,
        feature_name: str,
        default: float = 0.0,
    ) -> float:
        """指定した特徴量の加重平均変化量を取得する。"""
        return self.weighted_changes.get(
            feature_name,
            default,
        )


def compute_visit_weighted_change(
    variations: Sequence[MCTSVariation],
) -> F38VisitWeightedChange:
    """
    MCTS訪問回数を重みとして特徴量変化を集約する。

    Parameters
    ----------
    variations:
        MCTS探索系列の一覧。

    Returns
    -------
    F38VisitWeightedChange
        各特徴量の訪問回数加重平均変化量。
    """

    if not variations:
        return F38VisitWeightedChange(
            weighted_changes={}
        )

    total_visits = sum(
        variation.visits
        for variation in variations
    )

    if total_visits <= 0:
        return F38VisitWeightedChange(
            weighted_changes={}
        )

    feature_names = set()

    for variation in variations:
        feature_names.update(
            variation.deltas.keys()
        )

    weighted_changes: Dict[str, float] = {}

    for feature_name in sorted(feature_names):
        weighted_sum = 0.0

        for variation in variations:
            delta = variation.deltas.get(
                feature_name,
                0.0,
            )

            weighted_sum += (
                variation.visits * delta
            )

        weighted_changes[feature_name] = (
            weighted_sum / total_visits
        )

    return F38VisitWeightedChange(
        weighted_changes=weighted_changes
    )


# 互換用エイリアス
compute_mcts_visit_weighted_change = (
    compute_visit_weighted_change
)