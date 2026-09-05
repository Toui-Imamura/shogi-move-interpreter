"""
features/attack_defense.py

F32: 攻守切り替え

指し手によって攻撃・守備のバランスが
どちらの方向へ変化したかを表す特徴量。

F32:

    (ΔAttack - ΔDefense)
    --------------------
    |ΔAttack| + |ΔDefense| + ε

Attack:
    F19 攻撃圧力変化

Defense:
    F23 守備力変化

値域:
    -1 ～ +1

解釈:
    +1に近い:
        攻撃方向への変化

    0付近:
        攻守の変化が均衡、または変化が小さい

    -1に近い:
        守備方向への変化

注意:
    F32の値は「良い手・悪い手」を表すものではなく、
    攻守バランスの変化方向を表す。
"""

from __future__ import annotations

from dataclasses import dataclass

import cshogi

from .attack import f19_attack_pressure_change
from .defense import f23_defense_change
from .common import BLACK, WHITE


# ============================================================================
# Constants
# ============================================================================

EPSILON = 1e-8


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass(frozen=True)
class F32AttackDefenseSwitch:
    """
    F32 攻守切り替え。

    score:
        攻守切り替えの総合値

    attack_change:
        F19による攻撃圧力の変化

    defense_change:
        F23による守備力の変化
    """

    score: float
    attack_change: float
    defense_change: float


@dataclass(frozen=True)
class F32AttackDefenseSwitchChange:
    """
    F32の黒・白それぞれの値と、その差分。
    """

    black: float
    white: float
    difference: float


# ============================================================================
# Core calculation
# ============================================================================


def _attack_defense_switch(
    attack_change: float,
    defense_change: float,
    epsilon: float = EPSILON,
) -> float:
    """
    攻撃変化と守備変化からF32を計算する。

    F32:

        (ΔAttack - ΔDefense)
        --------------------
        |ΔAttack| + |ΔDefense| + ε

    Parameters
    ----------
    attack_change : float
        攻撃圧力の変化

    defense_change : float
        守備力の変化

    epsilon : float
        0除算防止用の微小値

    Returns
    -------
    float
        F32スコア
    """

    denominator = (
        abs(attack_change)
        + abs(defense_change)
        + epsilon
    )

    return (
        attack_change
        - defense_change
    ) / denominator


# ============================================================================
# Feature extraction
# ============================================================================


def f32_attack_defense_switch(
    before: cshogi.Board,
    after: cshogi.Board,
    color: int,
) -> F32AttackDefenseSwitch:
    """
    before -> after におけるF32を計算する。

    Parameters
    ----------
    before : cshogi.Board
        指し手前の局面

    after : cshogi.Board
        指し手後の局面

    color : int
        対象プレイヤー

    Returns
    -------
    F32AttackDefenseSwitch
        F32の値と構成要素
    """

    if color not in (BLACK, WHITE):
        raise ValueError(
            f"Invalid color: {color}"
        )

    # ------------------------------------------------------------------
    # F19: 攻撃圧力変化
    #
    # F19は黒・白の両方をまとめて返す。
    # ------------------------------------------------------------------

    attack = f19_attack_pressure_change(
        before,
        after,
    )

    if color == BLACK:
        attack_change = float(
            attack.black
        )
    else:
        attack_change = float(
            attack.white
        )

    # ------------------------------------------------------------------
    # F23: 守備力変化
    #
    # 現在のF23はcolorを指定して計算する。
    # ------------------------------------------------------------------

    defense = f23_defense_change(
        before,
        after,
        color,
    )

    defense_change = float(
        defense.score
    )

    # ------------------------------------------------------------------
    # F32
    # ------------------------------------------------------------------

    score = _attack_defense_switch(
        attack_change,
        defense_change,
    )

    return F32AttackDefenseSwitch(
        score=float(score),
        attack_change=attack_change,
        defense_change=defense_change,
    )


# ============================================================================
# Black / White
# ============================================================================


def f32_attack_defense_switch_change(
    before: cshogi.Board,
    after: cshogi.Board,
) -> F32AttackDefenseSwitchChange:
    """
    黒・白それぞれについてF32を計算し、
    黒から白を引いた差分を返す。
    """

    black = f32_attack_defense_switch(
        before,
        after,
        BLACK,
    )

    white = f32_attack_defense_switch(
        before,
        after,
        WHITE,
    )

    difference = (
        black.score
        - white.score
    )

    return F32AttackDefenseSwitchChange(
        black=float(black.score),
        white=float(white.score),
        difference=float(difference),
    )