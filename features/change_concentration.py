"""
F39: 特徴量変化の集中度

特徴量変化の絶対値を正規化し、
少数の特徴量に変化が集中しているかを表す。

p_i = |Δf_i| / Σ_j |Δf_j|

Concentration = Σ_i p_i^2

値が大きい:
    少数の特徴量に変化が集中

値が小さい:
    多数の特徴量に変化が分散
"""

from __future__ import annotations

from typing import Mapping


EPSILON = 1e-8


def compute_feature_change_concentration(
    feature_deltas: Mapping[str, float],
    epsilon: float = EPSILON,
) -> float:
    """
    全特徴量変化から変化集中度を計算する。

    Parameters
    ----------
    feature_deltas:
        特徴量名から変化量 Δf_i へのマッピング。

        例:
            {
                "F13": 0.8,
                "F14": 0.1,
                "F15": 0.0,
            }

    epsilon:
        全変化量が極めて小さい場合を判定するための閾値。

    Returns
    -------
    float
        特徴量変化の集中度。

        [0, 1] の範囲を取る。

        1.0:
            1つの特徴量に完全に集中

        値が小さい:
            多数の特徴量に分散

        全特徴量の変化が0の場合:
            0.0
    """

    if not feature_deltas:
        return 0.0

    absolute_changes = [
        abs(float(delta))
        for delta in feature_deltas.values()
    ]

    total_change = sum(absolute_changes)

    if total_change <= epsilon:
        return 0.0

    concentration = sum(
        (absolute_change / total_change) ** 2
        for absolute_change in absolute_changes
    )

    # 数値誤差による範囲外を防ぐ
    return max(0.0, min(1.0, concentration))