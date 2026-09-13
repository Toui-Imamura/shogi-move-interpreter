"""
tests/test_feature_adapters.py

feature_adapters.py の変化量変換を検証する。
"""

import cshogi

from features.attack import (
    f11_attack_pressure,
    f12_attackers,
)
from features.king_safety import (
    f26_king_safety,
)
from interpreter.feature_adapters import (
    compute_attack_feature_deltas,
    compute_defense_feature_deltas,
)


def make_after_board():
    """
    初期局面から合法手を1手進めた局面を作る。
    """
    board = cshogi.Board()

    move = board.move_from_usi("7g7f")
    assert move is not None
    assert board.is_legal(move)

    board.push(move)
    return board


def test_f11_adapter_returns_after_minus_before():
    before = cshogi.Board()
    after = make_after_board()

    result = compute_attack_feature_deltas(
        before,
        after,
    )

    expected = (
        f11_attack_pressure(after).difference
        - f11_attack_pressure(before).difference
    )

    assert result["F11"] == expected


def test_f12_adapter_returns_after_minus_before():
    before = cshogi.Board()
    after = make_after_board()

    result = compute_attack_feature_deltas(
        before,
        after,
    )

    expected = (
        f12_attackers(after).difference
        - f12_attackers(before).difference
    )

    assert result["F12"] == expected


def test_f26_adapter_returns_after_minus_before():
    before = cshogi.Board()
    after = make_after_board()

    result = compute_defense_feature_deltas(
        before,
        after,
    )

    expected = (
        f26_king_safety(after).difference
        - f26_king_safety(before).difference
    )

    assert result["F26"] == expected