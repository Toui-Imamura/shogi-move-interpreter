"""
F34: 戦略方向性変化

F13～F33の特徴量変化を5つの上位カテゴリへ集約する。

    Material
    Attack
    Defense
    Activity
    Formation

入力:
    feature_deltas: Dict[str, float]

出力:
    5方向の戦略ベクトルと、その変化量
"""

from dataclasses import dataclass
from typing import Dict


# ============================================================
# F34 feature groups
# ============================================================

MATERIAL_FEATURES = (
    "F13",
)

ATTACK_FEATURES = (
    "F14",
    "F15",
    "F19",
    "F20",
    "F21",
    "F22",
    "F32",
)

DEFENSE_FEATURES = (
    "F17",
    "F18",
    "F23",
    "F24",
    "F25",
    "F26",
    "F27",
    "F30",
)

ACTIVITY_FEATURES = (
    "F16",
)

FORMATION_FEATURES = (
    "F28",
    "F29",
    "F31",
    "F33",
)


FEATURE_GROUPS = {
    "Material": MATERIAL_FEATURES,
    "Attack": ATTACK_FEATURES,
    "Defense": DEFENSE_FEATURES,
    "Activity": ACTIVITY_FEATURES,
    "Formation": FORMATION_FEATURES,
}


@dataclass(frozen=True)
class F34StrategyDirection:
    """
    F34 戦略方向性変化。

    各値は、その戦略方向へどの程度変化したかを表す。
    """

    material: float
    attack: float
    defense: float
    activity: float
    formation: float

    @property
    def vector(self):
        return (
            self.material,
            self.attack,
            self.defense,
            self.activity,
            self.formation,
        )


def _group_delta(
    feature_deltas: Dict[str, float],
    features: tuple[str, ...],
) -> float:
    """
    指定された特徴量群の変化量を平均する。
    """

    values = [
        float(feature_deltas[name])
        for name in features
        if name in feature_deltas
    ]

    if not values:
        return 0.0

    return sum(values) / len(values)


def compute_strategy_direction(
    feature_deltas: Dict[str, float],
) -> F34StrategyDirection:
    """
    F13～F33の特徴量変化を5方向へ集約する。

    Parameters
    ----------
    feature_deltas:
        特徴量名をキー、現在局面から未来局面への変化量を値とする辞書。

        例:
            {
                "F13": 0.30,
                "F14": 0.20,
                "F15": 0.10,
                "F16": 0.05,
                "F28": -0.10,
            }

    Returns
    -------
    F34StrategyDirection
        Material / Attack / Defense /
        Activity / Formation の5方向の変化。
    """

    material = _group_delta(
        feature_deltas,
        MATERIAL_FEATURES,
    )

    attack = _group_delta(
        feature_deltas,
        ATTACK_FEATURES,
    )

    defense = _group_delta(
        feature_deltas,
        DEFENSE_FEATURES,
    )

    activity = _group_delta(
        feature_deltas,
        ACTIVITY_FEATURES,
    )

    formation = _group_delta(
        feature_deltas,
        FORMATION_FEATURES,
    )

    return F34StrategyDirection(
        material=material,
        attack=attack,
        defense=defense,
        activity=activity,
        formation=formation,
    )


def strategy_direction_change(
    feature_deltas: Dict[str, float],
) -> F34StrategyDirection:
    """
    F34の公開用エイリアス。

    compute_strategy_direction() と同じ計算を行う。
    """

    return compute_strategy_direction(feature_deltas)