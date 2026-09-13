"""
interpreter/pipeline_result_adapter.py

UnifiedFeaturePipelineResultを、
F01〜F40の共通辞書および固定長ベクトルへ変換する。

変換方針
--------
F01〜F33:
    対象手による即時変化量を使用する。

F34〜F36:
    Variationの分析結果を使用する。

F37:
    MCTS各特徴量の変化頻度の平均値を使用する。
    詳細な特徴量別頻度はMCTSFeatureResultに保持される。

F38:
    MCTS訪問重み付き変化の平均絶対値を使用する。
    詳細な特徴量別変化量はMCTSFeatureResultに保持される。

F39:
    MCTSの変化集中度を使用する。

F40:
    対象手による局面遷移度を使用する。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from interpreter.feature_vector import (
    FEATURE_NAMES,
    dict_to_vector,
)


def _mean(values: list[float]) -> float:
    """空配列では0、それ以外では算術平均を返す。"""
    if not values:
        return 0.0

    return sum(values) / len(values)


def _mean_absolute(values: list[float]) -> float:
    """値の絶対値の平均を返す。"""
    if not values:
        return 0.0

    return sum(abs(value) for value in values) / len(values)


def _extract_variation_values(
    variation_result: Any,
) -> dict[str, float]:
    """
    VariationFeatureResultからF34〜F36を取得する。
    """

    if variation_result is None:
        return {}

    variation_deltas = getattr(
        variation_result,
        "variation_deltas",
        {},
    )

    if not isinstance(variation_deltas, Mapping):
        return {}

    result: dict[str, float] = {}

    # --------------------------------------------------
    # F13・F15
    # --------------------------------------------------
    # F13: 駒得形成過程
    # F15: 攻撃継続性
    #
    # これらはVariationFeatureResultで計算されるため、
    # 即時特徴量ではなくVariation側から取得する。

    for name in ("F13", "F15"):
        if name in variation_deltas:
            result[name] = float(
                variation_deltas[name]
            )

    # --------------------------------------------------
    # F34〜F36
    # --------------------------------------------------

    for name in ("F35", "F36"):
        if name in variation_deltas:
            result[name] = float(
                variation_deltas[name]
            )

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
        result["F34"] = _mean(f34_values)

    return result


def _extract_mcts_values(
    mcts_result: Any,
) -> dict[str, float]:
    """
    MCTSFeatureResultからF37〜F39を取得する。
    """

    if mcts_result is None:
        return {}

    result: dict[str, float] = {}

    # F37: 特徴量別の変化頻度を平均して1つの値にする。
    f37 = getattr(mcts_result, "f37", None)

    if f37 is not None:
        frequencies = getattr(
            f37,
            "feature_frequencies",
            {},
        )

        if isinstance(frequencies, Mapping):
            result["F37"] = _mean(
                [
                    float(value)
                    for value in frequencies.values()
                ]
            )

    # F38: 特徴量別の訪問重み付き変化量の
    # 平均絶対値を1つの値にする。
    f38 = getattr(mcts_result, "f38", None)

    if f38 is not None:
        weighted_changes = getattr(
            f38,
            "weighted_changes",
            {},
        )

        if isinstance(weighted_changes, Mapping):
            result["F38"] = _mean_absolute(
                [
                    float(value)
                    for value in weighted_changes.values()
                ]
            )

    # F39: 変化集中度
    if hasattr(mcts_result, "f39"):
        result["F39"] = float(
            mcts_result.f39
        )

    return result


def pipeline_result_to_dict(
    result: Any,
    *,
    include_unimplemented_as_zero: bool = True,
) -> dict[str, float]:
    """
    UnifiedFeaturePipelineResultをF01〜F40の辞書へ変換する。

    Parameters
    ----------
    result:
        compute_feature_pipeline()の戻り値。

    include_unimplemented_as_zero:
        未設定の特徴量を0.0で補完するか。

    Returns
    -------
    dict[str, float]
        F01〜F40の特徴量辞書。
    """

    if result is None:
        raise TypeError(
            "result must not be None"
        )

    if not hasattr(result, "immediate_deltas"):
        raise TypeError(
            "resultにimmediate_deltas属性がありません。"
        )

    values: dict[str, float] = {}

    # --------------------------------------------------
    # F01〜F33
    # --------------------------------------------------

    immediate_deltas = result.immediate_deltas

    if isinstance(immediate_deltas, Mapping):
        for name, value in immediate_deltas.items():
            if name in FEATURE_NAMES:
                values[name] = float(value)

    # --------------------------------------------------
    # F34〜F36
    # --------------------------------------------------

    variation_values = _extract_variation_values(
        getattr(result, "variation", None)
    )
    values.update(variation_values)

    # --------------------------------------------------
    # F37〜F39
    # --------------------------------------------------

    mcts_values = _extract_mcts_values(
        getattr(result, "mcts", None)
    )
    values.update(mcts_values)

    # --------------------------------------------------
    # F40
    # --------------------------------------------------

    if hasattr(result, "f40_degree"):
        values["F40"] = float(
            result.f40_degree
        )
    elif hasattr(result, "f40"):
        f40 = result.f40

        if hasattr(f40, "degree"):
            values["F40"] = float(
                f40.degree
            )

    # --------------------------------------------------
    # 未実装・未計算の特徴量を0で補完
    # --------------------------------------------------

    if include_unimplemented_as_zero:
        for name in FEATURE_NAMES:
            values.setdefault(name, 0.0)

    return {
        name: float(values[name])
        for name in FEATURE_NAMES
        if name in values
    }


def pipeline_result_to_vector(
    result: Any,
    *,
    include_unimplemented_as_zero: bool = True,
) -> list[float]:
    """
    UnifiedFeaturePipelineResultを長さ40のベクトルへ変換する。
    """

    values = pipeline_result_to_dict(
        result,
        include_unimplemented_as_zero=(
            include_unimplemented_as_zero
        ),
    )

    return dict_to_vector(
        values,
        default=0.0,
        strict=True,
    )
