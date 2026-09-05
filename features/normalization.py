"""
features/normalization.py

特徴量のスケールを統一するための正規化処理。

基本形式:
    normalized = tanh(value / scale)

各特徴量について個別のscaleを設定できる。
"""

from __future__ import annotations

import math
from typing import Mapping


def tanh_normalize(
    value: float,
    scale: float,
) -> float:
    """
    tanh(value / scale) による正規化。

    戻り値は [-1, 1] の範囲に収まる。
    """

    if scale <= 0:
        raise ValueError(
            "scale must be greater than 0"
        )

    return math.tanh(
        float(value) / scale
    )


def normalize_feature_deltas(
    feature_deltas: Mapping[str, float],
    scales: Mapping[str, float],
) -> dict[str, float]:
    """
    特徴量ごとにtanh正規化を行う。

    scalesに存在しない特徴量は
    scale=1.0として扱う。
    """

    result: dict[str, float] = {}

    for feature_name, value in feature_deltas.items():

        scale = scales.get(
            feature_name,
            1.0,
        )

        result[feature_name] = tanh_normalize(
            value,
            scale,
        )

    return result