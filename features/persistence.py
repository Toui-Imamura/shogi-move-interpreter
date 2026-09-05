"""
F36: 変化持続性

MCTSの1本のVariationにおいて、
各特徴量の変化方向が最終局面までどの程度維持されたかを評価する。

S0 -> S1 -> ... -> Sv

各時点の特徴量変化:
    Δf_i,t = f_i(S_t) - f_i(S_{t-1})

最終変化:
    Δf_i,end = f_i(S_v) - f_i(S_0)

途中の変化方向と最終変化方向の符号が一致している割合を
変化持続性として計算する。

値の範囲:
    0.0 ～ 1.0
"""

from dataclasses import dataclass
from typing import Dict, Sequence

from .strategy import FEATURE_GROUPS


EPSILON = 1e-8


@dataclass(frozen=True)
class F36Persistence:
    """
    F36 変化持続性。
    """

    score: float
    feature_scores: Dict[str, float]

    def get(self, feature_name: str) -> float:
        """
        指定した特徴量の持続性を取得する。
        """
        return self.feature_scores.get(feature_name, 0.0)


def _sign(value: float, epsilon: float = EPSILON) -> int:
    """
    浮動小数点誤差を考慮して符号を取得する。
    """

    if value > epsilon:
        return 1

    if value < -epsilon:
        return -1

    return 0


def _persistence_score(
    deltas: Sequence[float],
) -> float:
    """
    1特徴量の変化持続性を計算する。

    Parameters
    ----------
    deltas:
        各ステップにおける特徴量変化。

    Returns
    -------
    float
        0.0～1.0
    """

    if not deltas:
        return 0.0

    final_sign = _sign(deltas[-1])

    # 最終的に変化していない場合、
    # 「どの方向へ変化したか」を定義できない。
    if final_sign == 0:
        return 0.0

    meaningful_deltas = [
        delta
        for delta in deltas
        if _sign(delta) != 0
    ]

    if not meaningful_deltas:
        return 0.0

    persistent = sum(
        1
        for delta in meaningful_deltas
        if _sign(delta) == final_sign
    )

    return persistent / len(meaningful_deltas)


def compute_feature_persistence(
    feature_deltas: Dict[str, Sequence[float]],
) -> F36Persistence:
    """
    複数特徴量の変化持続性を計算する。

    Parameters
    ----------
    feature_deltas:
        特徴量ごとの時系列変化。

        例:
            {
                "F13": [0.2, 0.1, 0.05],
                "F14": [0.1, -0.1, 0.2],
            }

    Returns
    -------
    F36Persistence
        各特徴量の持続性と全体スコア。
    """

    feature_scores = {
        feature_name: _persistence_score(deltas)
        for feature_name, deltas in feature_deltas.items()
    }

    if not feature_scores:
        return F36Persistence(
            score=0.0,
            feature_scores={},
        )

    score = sum(feature_scores.values()) / len(feature_scores)

    return F36Persistence(
        score=score,
        feature_scores=feature_scores,
    )


def compute_persistence(
    feature_deltas: Dict[str, Sequence[float]],
) -> F36Persistence:
    """
    compute_feature_persistence() の別名。
    """

    return compute_feature_persistence(feature_deltas)