"""
interpreter/feature_adapters.py

F01～F33の個別実装を、
指し手解釈システムで扱う共通形式へ変換する。

基本方針:
    before = 指し手前局面
    after  = 指し手後局面

    delta = after - before

F01～F33には、

    black / white / difference
    score
    change
    color指定

など異なる戻り値形式が存在するため、
このファイルで吸収する。
"""

from __future__ import annotations

from typing import Any, Mapping

import cshogi

from features.basic import (
    f01_material,
    f02_hand_material,
    f03_exchange,
    f04_major_balance,
    f05_minor_composition,
)

from features.activity import (
    f06_control,
    f07_mobility,
    f08_important_control,
    f09_idle_improvement,
    f10_activity,
    f16_advancement,
    f17_defense_pieces,
    f18_defense_control,
)

from features.attack import (
    f11_attack_pressure,
    f12_attackers,
    f13_material_gain_process,
    f14_attack_base,
    f15_attack_continuity,
    f19_attack_pressure_change,
    f20_attackers_change,
    f21_attack_concentration_change,
    f22_threat_formation_change,
)

from features.defense import (
    f23_defense_change,
)

from features.king_safety import (
    f24_king_safety_change,
    f26_king_safety,
)

from features.response import (
    f25_response_change,
)

from features.defensive_placement import (
    f27_defensive_placement,
    f27_defensive_placement_change,
)

from features.formation import (
    f28_formation,
    f28_formation_change,
)

from features.castle import (
    f29_castle_progress,
    f29_castle_progress_change,
)

from features.weakness import (
    f30_weakness,
    f30_weakness_change,
)

from features.pawn_structure import (
    f31_pawn_connectivity,
    f31_pawn_connectivity_change,
)

from features.attack_defense import (
    f32_attack_defense_switch,
)

from features.force_distribution import (
    f33_force_distribution,
)


BLACK = cshogi.BLACK
WHITE = cshogi.WHITE


def _difference(value: Any) -> float:
    """
    .difference を持つ結果から差分値を取得する。
    """
    return float(value.difference)


def _score(value: Any) -> float:
    """
    .score を持つ結果からスカラー値を取得する。
    """
    return float(value.score)


def _color_score(
    value: Any,
    color: int,
) -> float:
    """
    black / white を持つ結果から指定色の値を取得する。
    """
    if color == BLACK:
        return float(value.black)

    return float(value.white)


def _color_delta(
    before: Any,
    after: Any,
    color: int,
) -> float:
    """
    black / white の局面評価から
    指定色の変化量を計算する。
    """
    return _color_score(after, color) - _color_score(before, color)


def _difference_delta(
    before: Any,
    after: Any,
) -> float:
    """
    difference の局面評価から変化量を計算する。
    """
    return _difference(after) - _difference(before)


def compute_basic_feature_deltas(
    before,
    after,
    move=None,
) -> dict[str, float]:
    """
    F01～F05を計算する。

    F03は実際の指し手が必要なため、
    moveが指定されている場合のみ計算する。
    """

    result: dict[str, float] = {}

    result["F01"] = _difference_delta(
        f01_material(before),
        f01_material(after),
    )

    result["F02"] = _difference_delta(
        f02_hand_material(before),
        f02_hand_material(after),
    )

    result["F04"] = _difference_delta(
        f04_major_balance(before),
        f04_major_balance(after),
    )

    # F05のdifferenceは駒種ごとの辞書。
    # 各駒種の構成差の変化量を合計して、
    # 小駒構成の総合的な変化量として扱う。
    f05_before = f05_minor_composition(before)
    f05_after = f05_minor_composition(after)

    piece_types = set(f05_before.difference) | set(
        f05_after.difference
    )

    f05_delta = sum(
        abs(
            float(
                f05_after.difference.get(
                    piece_type,
                    0,
                )
            )
            - float(
                f05_before.difference.get(
                    piece_type,
                    0,
                )
            )
        )
        for piece_type in piece_types
    )

    result["F05"] = float(f05_delta)

    if move is not None:
        result["F03"] = float(
                f03_exchange(
                    before,
                    after,
                    move,
                ).score
        )

    return result


def compute_activity_feature_deltas(
    before,
    after,
) -> dict[str, float]:
    """
    F06～F10, F16～F18を計算する。
    """

    result: dict[str, float] = {}

    # 局面そのものから計算する特徴量
    state_features = {
        "F06": f06_control,
        "F07": f07_mobility,
        "F08": f08_important_control,
        "F10": f10_activity,
        "F16": f16_advancement,
        "F17": f17_defense_pieces,
        "F18": f18_defense_control,
    }

    for name, extractor in state_features.items():
        before_value = extractor(before)
        after_value = extractor(after)

        result[name] = _difference_delta(
            before_value,
            after_value,
        )

    # F09はbefore/afterを直接受け取る特徴量
    result["F09"] = _difference(
        f09_idle_improvement(
            before,
            after,
        )
    )

    return result

def compute_attack_feature_deltas(
    before,
    after,
    variation=None,
) -> dict[str, float]:
    """
    F11～F15, F19～F22を計算する。

    F13/F15はVariationを必要とするため、
    variationが与えられた場合に計算する。
    """

    result: dict[str, float] = {}

    result["F11"] = _difference_delta(
        f11_attack_pressure(before),
        f11_attack_pressure(after),
    )

    result["F12"] = _difference_delta(
        f12_attackers(before),
        f12_attackers(after),
    )

    if variation is not None:
        result["F13"] = _difference(
            f13_material_gain_process(variation)
        )

        result["F15"] = _difference(
            f15_attack_continuity(variation)
        )

    result["F14"] = _difference(
        f14_attack_base(
            before,
            after,
        )
    )

    f19 = f19_attack_pressure_change(
        before,
        after,
    )
    result["F19"] = _difference(f19)

    f20 = f20_attackers_change(
        before,
        after,
    )
    result["F20"] = _difference(f20)

    result["F21"] = float(
        f21_attack_concentration_change(
            before,
            after,
            BLACK,
        ).change
    )

    f22_black = f22_threat_formation_change(
        before,
        after,
        BLACK,
    )

    f22_white = f22_threat_formation_change(
        before,
        after,
        WHITE,
    )

    result["F22"] = float(
        f22_black.change - f22_white.change
    )
    return result


def compute_defense_feature_deltas(
    before,
    after,
) -> dict[str, float]:
    """
    F23～F30を計算する。
    """

    result: dict[str, float] = {}

    # F23
    f23_black = f23_defense_change(
        before,
        after,
        BLACK,
    )

    f23_white = f23_defense_change(
        before,
        after,
        WHITE,
    )

    result["F23"] = float(
        f23_black.score - f23_white.score
    )

    # F24
    result["F24"] = _difference(
        f24_king_safety_change(
            before,
            after,
        )
    )

    # F25
    f25_black = f25_response_change(
        before,
        after,
        BLACK,
    )

    f25_white = f25_response_change(
        before,
        after,
        WHITE,
    )

    result["F25"] = float(
        f25_black.score - f25_white.score
    )

    # F26
    result["F26"] = _difference_delta(
        f26_king_safety(before),
        f26_king_safety(after),
    )

    # F27
    result["F27"] = _difference(
        f27_defensive_placement_change(
            before,
            after,
        )
    )

    # F28
    result["F28"] = _difference(
        f28_formation_change(
            before,
            after,
        )
    )

    # F29
    result["F29"] = _difference(
        f29_castle_progress_change(
            before,
            after,
        )
    )

    # F30
    result["F30"] = _difference(
        f30_weakness_change(
            before,
            after,
        )
    )

    return result


def compute_structure_feature_deltas(
    before,
    after,
) -> dict[str, float]:
    """
    F31～F33を計算する。
    """

    result: dict[str, float] = {}

    result["F31"] = _difference(
        f31_pawn_connectivity_change(
            before,
            after,
        )
    )

    f32_black = f32_attack_defense_switch(
        before,
        after,
        BLACK,
    )

    f32_white = f32_attack_defense_switch(
        before,
        after,
        WHITE,
    )

    result["F32"] = float(
        f32_black.score - f32_white.score
    )

    f33_black = f33_force_distribution(
        after,
        BLACK,
    )

    f33_black_before = f33_force_distribution(
        before,
        BLACK,
    )

    f33_white = f33_force_distribution(
        after,
        WHITE,
    )

    f33_white_before = f33_force_distribution(
        before,
        WHITE,
    )

    black_delta = (
        f33_black.score
        - f33_black_before.score
    )

    white_delta = (
        f33_white.score
        - f33_white_before.score
    )

    result["F33"] = float(
        black_delta - white_delta
    )

    return result


def compute_f01_f33_deltas(
    before,
    after,
    move=None,
    variation=None,
) -> dict[str, float]:
    """
    F01～F33をまとめて計算する。

    Parameters
    ----------
    before:
        指し手前のcshogi.Board

    after:
        指し手後のcshogi.Board

    move:
        F03で使用するcshogiの指し手。
        Noneでも計算可能だが、その場合F03は含まれない。

    variation:
        F13/F15で使用するVariation。
        Noneの場合、それらは含まれない。
    """

    if before is None:
        raise ValueError("before must not be None")

    if after is None:
        raise ValueError("after must not be None")

    result: dict[str, float] = {}

    result.update(
        compute_basic_feature_deltas(
            before,
            after,
            move,
        )
    )

    result.update(
        compute_activity_feature_deltas(
            before,
            after,
        )
    )

    result.update(
        compute_attack_feature_deltas(
            before,
            after,
            variation,
        )
    )

    result.update(
        compute_defense_feature_deltas(
            before,
            after,
        )
    )

    result.update(
        compute_structure_feature_deltas(
            before,
            after,
        )
    )

    return dict(sorted(result.items()))
