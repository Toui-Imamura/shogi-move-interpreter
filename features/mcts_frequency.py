"""
F37: MCTS変化頻度

複数のMCTS探索系列において、
特定の特徴量変化が何本の系列で観測されたかを求める。

Frequency_i =
    # variations with change_i / # variations

注意:
    この頻度は「指し手の意図の確率」ではない。
    複数の探索系列で繰り返し観測された変化の割合を表す。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence


EPSILON = 1e-8


@dataclass(frozen=True)
class F37MCTSChangeFrequency:
    """
    F37 MCTS変化頻度。

    feature_frequencies:
        特徴量名 -> その特徴量の変化が観測された探索系列の割合
    """

    feature_frequencies: Dict[str, float]

    def get(self, feature_name: str, default: float = 0.0) -> float:
        """指定した特徴量の頻度を取得する。"""
        return self.feature_frequencies.get(feature_name, default)


def _has_change(delta: float, epsilon: float = EPSILON) -> bool:
    """
    特徴量変化が観測されたか判定する。

    浮動小数点誤差を考慮し、
    絶対値がepsilonより大きい場合を変化ありとする。
    """
    return abs(delta) > epsilon


def compute_mcts_change_frequency(
    variations: Sequence[Mapping[str, float]],
    epsilon: float = EPSILON,
) -> F37MCTSChangeFrequency:
    """
    複数のMCTS探索系列から特徴量変化頻度を計算する。

    Parameters
    ----------
    variations:
        MCTS探索系列ごとの特徴量変化。

        例:
        [
            {"F13": 0.2, "F14": 0.0},
            {"F13": 0.1, "F14": -0.1},
            {"F13": 0.0, "F14": -0.2},
        ]

        これは3本の探索系列を表す。

    epsilon:
        変化ありと判定する最小絶対値。

    Returns
    -------
    F37MCTSChangeFrequency
        各特徴量について、
        「変化が観測された探索系列数 / 全探索系列数」
        を格納する。
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    total_variations = len(variations)

    if total_variations == 0:
        return F37MCTSChangeFrequency(feature_frequencies={})

    # 全探索系列に登場する特徴量を収集する。
    feature_names = set()

    for variation in variations:
        feature_names.update(variation.keys())

    frequencies: Dict[str, float] = {}

    for feature_name in sorted(feature_names):
        changed_count = 0

        for variation in variations:
            delta = variation.get(feature_name, 0.0)

            if _has_change(delta, epsilon):
                changed_count += 1

        frequencies[feature_name] = (
            changed_count / total_variations
        )

    return F37MCTSChangeFrequency(
        feature_frequencies=frequencies
    )


# 別名として利用可能にしておく。
compute_change_frequency = compute_mcts_change_frequency