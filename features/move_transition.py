"""
F40: 指し手による局面転換度

複数特徴量の変化量を統合し、
指し手によって局面の性質がどの程度変化したかを表す。

TransitionDegree = sqrt(sum_i lambda_i * (Delta f_i)^2)

注意:
    TransitionDegree は局面変化の大きさを表す指標であり、
    指し手の良し悪しを表すものではない。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping, Any


EPSILON = 1e-8


@dataclass(frozen=True)
class F40MoveTransition:
    """
    F40 指し手による局面転換度。

    Parameters
    ----------
    degree:
        局面転換度。

    feature_deltas:
        TransitionDegree の計算に使用した特徴量変化。
    """

    degree: float
    feature_deltas: Mapping[str, float]

    def get(self, feature_name: str, default: float = 0.0) -> float:
        """指定した特徴量の変化量を取得する。"""
        return float(self.feature_deltas.get(feature_name, default))


def _validate_position_pair(
    current: Any,
    future: Any,
) -> None:
    """
    current と future が指定されていることを確認する。

    F40の数値計算自体は feature_deltas に基づくため、
    現時点では局面オブジェクトの具体的な型には依存しない。
    """

    if current is None:
        raise ValueError("current position must not be None")

    if future is None:
        raise ValueError("future position must not be None")


def _transition_degree(
    feature_deltas: Mapping[str, float],
    feature_weights: Mapping[str, float] | None = None,
) -> float:
    """
    特徴量変化から局面転換度を計算する。

    TransitionDegree = sqrt(sum_i lambda_i * delta_i^2)

    feature_weights が指定されない場合、
    すべての特徴量の重みを1.0として扱う。
    """

    if not feature_deltas:
        return 0.0

    degree_squared = 0.0

    for feature_name, delta in feature_deltas.items():
        delta = float(delta)

        if feature_weights is None:
            weight = 1.0
        else:
            weight = float(feature_weights.get(feature_name, 1.0))

        if weight < 0.0:
            raise ValueError(
                f"feature weight must be non-negative: "
                f"{feature_name}={weight}"
            )

        degree_squared += weight * (delta ** 2)

    return sqrt(max(0.0, degree_squared))


def compute_move_transition_degree(
    current: Any,
    future: Any,
    feature_deltas: Mapping[str, float],
    feature_weights: Mapping[str, float] | None = None,
) -> float:
    """
    指し手による局面転換度を計算する。

    Parameters
    ----------
    current:
        指し手前の局面。

    future:
        MCTSなどによる未来局面。

    feature_deltas:
        指し手前後、または現在局面と未来局面の
        各特徴量の変化量。

        例:
            {
                "F13": 0.4,
                "F14": 0.2,
                "F15": -0.1,
            }

    feature_weights:
        各特徴量の重み lambda_i。
        指定しない場合はすべて1.0。

    Returns
    -------
    float
        指し手による局面転換度。

    Notes
    -----
    値が大きいことは「良い手」を意味しない。
    あくまで局面の性質がどの程度変化したかを表す。
    """

    _validate_position_pair(current, future)

    return _transition_degree(
        feature_deltas,
        feature_weights,
    )


def compute_move_transition(
    current: Any,
    future: Any,
    feature_deltas: Mapping[str, float],
    feature_weights: Mapping[str, float] | None = None,
) -> F40MoveTransition:
    """
    F40の詳細結果を取得する。

    compute_move_transition_degree() と同じ計算を行い、
    特徴量変化も一緒に保持する。
    """

    degree = compute_move_transition_degree(
        current,
        future,
        feature_deltas,
        feature_weights,
    )

    return F40MoveTransition(
        degree=degree,
        feature_deltas=dict(feature_deltas),
    )