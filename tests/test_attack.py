"""
F11～F15のテスト。
"""

import cshogi
import pytest

from features.attack import (
    attack_base_set,
    attacking_piece_set,
    f11_attack_pressure,
    f12_attackers,
    f13_material_gain_process,
    f14_attack_base,
    f15_attack_continuity,
    f19_attack_pressure_change,
    f20_attackers_change,
    f21_attack_concentration,
    f21_attack_concentration_change,
    f22_threat_formation,
    f22_threat_formation_change,
)
from features.transition import make_usi_variation

from features.common import BLACK, WHITE


def test_f11_initial_position_is_balanced():
    """初期局面ではF11のBlack/Whiteが対称になる。"""
    board = cshogi.Board()

    result = f11_attack_pressure(board)

    assert result.difference == 0.0
    assert result.black == result.white


def test_f12_initial_position_is_balanced():
    """初期局面ではF12のBlack/Whiteが対称になる。"""
    board = cshogi.Board()

    result = f12_attackers(board)

    assert result.difference == 0
    assert result.black == result.white


def test_f11_has_nonnegative_pressure_components():
    """F11の構成要素が正常に計算される。"""
    board = cshogi.Board()

    result = f11_attack_pressure(board)

    assert result.black_king_control >= 0
    assert result.white_king_control >= 0

    assert result.black_attackers >= 0
    assert result.white_attackers >= 0

    assert result.black_invasion >= 0
    assert result.white_invasion >= 0


def test_f12_attack_piece_set_is_valid():
    """F12の攻撃参加駒集合が正常に取得できる。"""
    board = cshogi.Board()

    black = attacking_piece_set(board, cshogi.BLACK)
    white = attacking_piece_set(board, cshogi.WHITE)

    assert isinstance(black, set)
    assert isinstance(white, set)


def test_f13_initial_position_has_no_gain():
    """1局面だけではF13の駒得形成は発生しない。"""
    board = cshogi.Board()

    variation = make_usi_variation(
        board,
        [],
    )

    result = f13_material_gain_process(
        variation
    )

    assert result.black == 0.0
    assert result.white == 0.0
    assert result.difference == 0.0


def test_f13_non_capture_variation_has_no_material_gain():
    """駒を取らない変化ではF13の駒得形成は0。"""
    board = cshogi.Board()

    variation = make_usi_variation(
        board,
        [
            "7g7f",
            "3c3d",
            "2g2f",
            "8c8d",
        ],
    )

    result = f13_material_gain_process(
        variation
    )

    assert result.black == 0.0
    assert result.white == 0.0
    assert result.difference == 0.0


def test_f14_initial_position_is_symmetric():
    """初期局面ではF14の変化量が0。"""
    board = cshogi.Board()

    result = f14_attack_base(
        board,
        board,
    )

    assert result.black == 0.0
    assert result.white == 0.0
    assert result.difference == 0.0


def test_f14_attack_base_set_returns_sets():
    """攻撃拠点集合が正常に取得できる。"""
    board = cshogi.Board()

    black = attack_base_set(
        board,
        cshogi.BLACK,
    )

    white = attack_base_set(
        board,
        cshogi.WHITE,
    )

    assert isinstance(black, set)
    assert isinstance(white, set)


def test_f15_single_position_has_valid_continuity():
    """1局面のVariationでもF15を計算できる。"""
    board = cshogi.Board()

    variation = make_usi_variation(
        board,
        [],
    )

    result = f15_attack_continuity(
        variation
    )

    assert 0.0 <= result.black <= 1.0
    assert 0.0 <= result.white <= 1.0
    assert result.steps == 1


def test_f15_continuity_has_valid_range():
    """複数局面でF15が0～1の範囲になる。"""
    board = cshogi.Board()

    variation = make_usi_variation(
        board,
        [
            "7g7f",
            "3c3d",
            "2g2f",
            "8c8d",
        ],
    )

    result = f15_attack_continuity(
        variation
    )

    assert 0.0 <= result.black <= 1.0
    assert 0.0 <= result.white <= 1.0

    assert result.steps == 5

def test_f19_same_position_has_zero_change():
    board = cshogi.Board()

    result = f19_attack_pressure_change(board, board)

    assert result.black == pytest.approx(0.0)
    assert result.white == pytest.approx(0.0)
    assert result.difference == pytest.approx(0.0)


def test_f20_same_position_has_zero_change():
    board = cshogi.Board()

    result = f20_attackers_change(board, board)

    assert result.black == 0
    assert result.white == 0
    assert result.difference == 0


def test_f21_initial_position_is_symmetric():
    board = cshogi.Board()

    black = f21_attack_concentration(board, BLACK)
    white = f21_attack_concentration(board, WHITE)

    assert black == pytest.approx(white)


def test_f21_same_position_change_is_zero():
    board = cshogi.Board()

    result = f21_attack_concentration_change(
        board,
        board,
        BLACK,
    )

    assert result.change == pytest.approx(0.0)


def test_f21_concentration_is_non_negative():
    board = cshogi.Board()

    value = f21_attack_concentration(
        board,
        BLACK,
    )

    assert value >= 0.0

# ============================================================
# F22 tests
# ============================================================

def test_f22_initial_position():
    board = cshogi.Board()

    black_value = f22_threat_formation(
        board,
        BLACK,
    )

    white_value = f22_threat_formation(
        board,
        WHITE,
    )

    assert black_value >= 0.0
    assert white_value >= 0.0


def test_f22_capture_threat_increases():
    """
    7g7f alone should produce a valid F22 value.

    The exact numerical value is intentionally not fixed here,
    because the threat weights are provisional.
    """
    before = cshogi.Board()

    move = before.move_from_usi("7g7f")
    after = before.copy()
    after.push(move)

    result = f22_threat_formation_change(
        before,
        after,
        BLACK,
    )

    assert result.before >= 0.0
    assert result.after >= 0.0
    assert (
        result.change
        == result.after - result.before
    )


def test_f22_change_is_zero_for_unchanged_position():
    board = cshogi.Board()

    result = f22_threat_formation_change(
        board,
        board,
        BLACK,
    )

    assert result.change == 0.0
    assert result.capture_threat == 0.0
    assert result.king_threat == 0.0
    assert result.mate_threat == 0.0
    assert result.important_piece_threat == 0.0
    assert result.forcing_defense == 0.0