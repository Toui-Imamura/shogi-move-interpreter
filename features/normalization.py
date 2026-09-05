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

# ============================================================
# F01〜F33 normalization scales
# ============================================================

DEFAULT_FEATURE_SCALES: dict[str, float] = {
    "F01": 1.0,
    "F02": 1.0,
    "F03": 1.0,
    "F04": 1.0,
    "F05": 1.0,
    "F06": 10.0,
    "F07": 10.0,
    "F08": 5.0,
    "F09": 1.0,
    "F10": 1.0,
    "F11": 1.0,
    "F12": 5.0,
    "F14": 1.0,
    "F16": 1.0,
    "F17": 5.0,
    "F18": 5.0,
    "F19": 1.0,
    "F20": 5.0,
    "F21": 1.0,
    "F22": 1.0,
    "F23": 1.0,
    "F24": 1.0,
    "F25": 1.0,
    "F26": 1.0,
    "F27": 1.0,
    "F28": 1.0,
    "F29": 1.0,
    "F30": 1.0,
    "F31": 1.0,
    "F32": 1.0,
    "F33": 1.0,
}

def normalize_default_feature_deltas(
    feature_deltas: Mapping[str, float],
) -> dict[str, float]:
    """
    F01〜F33の特徴量変化量を、
    暫定的なデフォルトscaleで正規化する。

    注意:
        DEFAULT_FEATURE_SCALESは暫定値であり、
        大量棋譜による統計分析後に更新する。
    """

    return normalize_feature_deltas(
        feature_deltas,
        DEFAULT_FEATURE_SCALES,
    )