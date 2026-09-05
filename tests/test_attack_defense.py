"""
tests/test_attack_defense.py

F32 攻守切り替えのテスト
"""

import math

import cshogi

from features.attack_defense import (
    _attack_defense_switch,
    f32_attack_defense_switch,
    f32_attack_defense_switch_change,
)
from features.common import BLACK, WHITE


# ============================================================================
# Helper
# ============================================================================


def apply_moves(
    board: cshogi.Board,
    moves: list[str],
) -> cshogi.Board:
    """
    USI形式の指し手を順番に適用する。
    """

    for move_usi in moves:
        move = board.move_from_usi(
            move_usi
        )

        assert board.is_legal(
            move
        ), f"Illegal move: {move_usi}"

        board.push(move)

    return board


# ============================================================================
# Formula tests
# ============================================================================


def test_attack_only():
    """
    攻撃変化のみの場合、F32は+1に近づく。
    """

    score = _attack_defense_switch(
        attack_change=1.0,
        defense_change=0.0,
    )

    assert math.isclose(
        score,
        1.0,
        rel_tol=1e-6,
    )


def test_defense_only():
    """
    守備変化のみの場合、F32は-1に近づく。
    """

    score = _attack_defense_switch(
        attack_change=0.0,
        defense_change=1.0,
    )

    assert math.isclose(
        score,
        -1.0,
        rel_tol=1e-6,
    )


def test_balanced_change():
    """
    攻撃変化と守備変化が同じ場合、
    F32は0付近になる。
    """

    score = _attack_defense_switch(
        attack_change=1.0,
        defense_change=1.0,
    )

    assert math.isclose(
        score,
        0.0,
        abs_tol=1e-6,
    )


def test_no_change():
    """
    攻撃・守備の変化がともに0の場合、
    F32は0になる。
    """

    score = _attack_defense_switch(
        attack_change=0.0,
        defense_change=0.0,
    )

    assert math.isclose(
        score,
        0.0,
        abs_tol=1e-6,
    )


def test_range():
    """
    F32が-1～+1の範囲に収まる。
    """

    values = [
        (-2.0, -1.0),
        (-1.0, 2.0),
        (0.5, -0.5),
        (1.0, 3.0),
        (3.0, 1.0),
        (10.0, 0.1),
    ]

    for attack, defense in values:
        score = _attack_defense_switch(
            attack,
            defense,
        )

        assert -1.0 <= score <= 1.0


# ============================================================================
# Position tests
# ============================================================================


def test_initial_position():
    """
    同一局面同士では変化がないため、
    F32は0になる。
    """

    before = cshogi.Board()
    after = cshogi.Board()

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

    assert math.isclose(
        black.score,
        0.0,
        abs_tol=1e-6,
    )

    assert math.isclose(
        white.score,
        0.0,
        abs_tol=1e-6,
    )


def test_actual_move():
    """
    実際の指し手による局面変化で
    F32が正常に計算できる。
    """

    before = cshogi.Board()

    after = cshogi.Board()

    after = apply_moves(
        after,
        [
            "7g7f",
        ],
    )

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

    assert isinstance(
        black.score,
        float,
    )

    assert isinstance(
        black.attack_change,
        float,
    )

    assert isinstance(
        black.defense_change,
        float,
    )

    assert isinstance(
        white.score,
        float,
    )

    assert -1.0 <= black.score <= 1.0
    assert -1.0 <= white.score <= 1.0


def test_black_white_change():
    """
    黒・白・差分を取得できる。
    """

    before = cshogi.Board()

    after = cshogi.Board()

    after = apply_moves(
        after,
        [
            "7g7f",
        ],
    )

    result = (
        f32_attack_defense_switch_change(
            before,
            after,
        )
    )

    assert isinstance(
        result.black,
        float,
    )

    assert isinstance(
        result.white,
        float,
    )

    assert isinstance(
        result.difference,
        float,
    )

    assert (
        math.isclose(
            result.difference,
            result.black - result.white,
            abs_tol=1e-6,
        )
    )


def test_invalid_color():
    """
    不正なcolorを指定するとValueErrorになる。
    """

    before = cshogi.Board()
    after = cshogi.Board()

    try:
        f32_attack_defense_switch(
            before,
            after,
            999,
        )
    except ValueError:
        return

    assert False, "ValueError was not raised"