"""
F35: 複数特徴量整合性

複数の特徴量変化が同じ方向を示しているかを評価する。

F34で集約した5つの戦略方向：

    Material
    Attack
    Defense
    Activity
    Formation

について、各方向の変化量の符号がどの程度一致しているかを
整合性として表現する。

値の範囲:
    0.0 ～ 1.0

1.0:
    複数の特徴量が強く同じ方向を示している

0.0:
    特徴量間で方向が一致していない
"""

from dataclasses import dataclass
from typing import Dict

from .strategy import compute_strategy_direction


EPSILON = 1e-8


@dataclass(frozen=True)
class F35Consistency:
    """
    F35 複数特徴量整合性。
    """

    score: float
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


def _sign(value: float, epsilon: float = EPSILON) -> int:
    """
    小さな値を0として符号を取得する。
    """

    if value > epsilon:
        return 1

    if value < -epsilon:
        return -1

    return 0


def _pairwise_consistency(values: tuple[float, ...]) -> float:
    """
    戦略方向間の符号整合性を計算する。

    非ゼロの方向のみを比較する。

    同方向:
        +1

    逆方向:
        0

    0:
        比較対象から除外
    """

    signs = [_sign(value) for value in values]
    signs = [sign for sign in signs if sign != 0]

    if len(signs) <= 1:
        return 1.0

    total = 0
    consistent = 0

    for i in range(len(signs)):
        for j in range(i + 1, len(signs)):
            total += 1

            if signs[i] == signs[j]:
                consistent += 1

    if total == 0:
        return 1.0

    return consistent / total


def compute_consistency(
    feature_deltas: Dict[str, float],
) -> F35Consistency:
    """
    F13～F33の特徴量変化からF35を計算する。

    Parameters
    ----------
    feature_deltas:
        特徴量名をキー、現在局面から未来局面への変化量を
        値とする辞書。

    Returns
    -------
    F35Consistency
        5方向の変化と整合性スコア。
    """

    strategy = compute_strategy_direction(feature_deltas)

    vector = strategy.vector

    score = _pairwise_consistency(vector)

    return F35Consistency(
        score=score,
        material=strategy.material,
        attack=strategy.attack,
        defense=strategy.defense,
        activity=strategy.activity,
        formation=strategy.formation,
    )


def compute_feature_consistency(
    feature_deltas: Dict[str, float],
) -> F35Consistency:
    """
    compute_consistency() の別名。
    """

    return compute_consistency(feature_deltas)