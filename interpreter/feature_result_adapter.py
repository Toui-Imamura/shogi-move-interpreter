"""
interpreter/feature_result_adapter.py

VariationFeatureResultなどの計算結果を、
F01〜F40の共通辞書および固定長ベクトルへ変換する。

現段階では:
    F01〜F33:
        transition_deltasの平均値

    F13/F15/F35/F36:
        variation_deltasから取得

    F34:
        F34_materialなどの各成分を平均

    F37〜F40:
        MCTS未接続のため0.0

という暫定集約を行う。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from interpreter.feature_vector import (
    FEATURE_NAMES,
    dict_to_vector,
)


def mean_transition_deltas(
    transition_deltas: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    """
    複数の遷移特徴量を特徴量ごとに平均する。
    """
    if not transition_deltas:
        return {}

    names: set[str] = set()

    for transition in transition_deltas:
        names.update(transition.keys())

    result: dict[str, float] = {}

    for name in names:
        values = [
            float(transition[name])
            for transition in transition_deltas
            if name in transition
        ]

        if values:
            result[name] = sum(values) / len(values)

    return result


def extract_variation_deltas(
    variation_deltas: Mapping[str, float],
) -> dict[str, float]:
    """
    Variation全体の特徴量を抽出する。
    """
    result: dict[str, float] = {}

    for name in ("F13", "F15", "F35", "F36"):
        if name in variation_deltas:
            result[name] = float(variation_deltas[name])

    # F34は複数の方向成分から構成されているため、
    # 現段階では平均値を代表値とする。
    f34_names = (
        "F34_material",
        "F34_attack",
        "F34_defense",
        "F34_activity",
        "F34_formation",
    )

    f34_values = [
        float(variation_deltas[name])
        for name in f34_names
        if name in variation_deltas
    ]

    if f34_values:
        result["F34"] = sum(f34_values) / len(f34_values)

    return result


def variation_result_to_dict(
    result: Any,
    *,
    include_unimplemented_as_zero: bool = True,
) -> dict[str, float]:
    """
    VariationFeatureResultをF01〜F40の辞書へ変換する。
    """
    if not hasattr(result, "transition_deltas"):
        raise TypeError(
            "resultにtransition_deltas属性がありません。"
        )

    if not hasattr(result, "variation_deltas"):
        raise TypeError(
            "resultにvariation_deltas属性がありません。"
        )

    values: dict[str, float] = {}

    values.update(
        mean_transition_deltas(
            result.transition_deltas
        )
    )

    values.update(
        extract_variation_deltas(
            result.variation_deltas
        )
    )

    if include_unimplemented_as_zero:
        for name in FEATURE_NAMES:
            values.setdefault(name, 0.0)

    return {
        name: float(values[name])
        for name in FEATURE_NAMES
        if name in values
    }


def variation_result_to_vector(
    result: Any,
    *,
    include_unimplemented_as_zero: bool = True,
) -> list[float]:
    """
    VariationFeatureResultを40次元ベクトルへ変換する。
    """
    values = variation_result_to_dict(
        result,
        include_unimplemented_as_zero=include_unimplemented_as_zero,
    )

    return dict_to_vector(
        values,
        default=0.0,
        strict=True,
    )
