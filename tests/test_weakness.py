import cshogi

from features.weakness import (
    attack_concentration_score,
    defense_thinness_score,
    f30_weakness,
    f30_weakness_change,
    invasion_risk_score,
)
from features.transition import apply_usi_move


def test_initial_weakness_is_valid():
    board = cshogi.Board()

    black = f30_weakness(
        board,
        cshogi.BLACK,
    )

    white = f30_weakness(
        board,
        cshogi.WHITE,
    )

    assert 0.0 <= black.score <= 1.0
    assert 0.0 <= white.score <= 1.0


def test_initial_black_white_are_symmetric():
    board = cshogi.Board()

    black = f30_weakness(
        board,
        cshogi.BLACK,
    )

    white = f30_weakness(
        board,
        cshogi.WHITE,
    )

    assert abs(
        black.score - white.score
    ) < 1e-9


def test_components_are_non_negative():
    board = cshogi.Board()

    result = f30_weakness(
        board,
        cshogi.BLACK,
    )

    assert result.defense_thinness >= 0.0
    assert result.attack_concentration >= 0.0
    assert result.invasion_risk >= 0.0


def test_same_position_change_is_zero():
    board = cshogi.Board()

    result = f30_weakness_change(
        board,
        board,
    )

    assert abs(result.black) < 1e-9
    assert abs(result.white) < 1e-9
    assert abs(result.difference) < 1e-9


def test_move_changes_weakness():
    board = cshogi.Board()

    before = board.copy()

    board = apply_usi_move(
        board,
        "7g7f",
    )

    result = f30_weakness_change(
        before,
        board,
    )

    assert (
        result.black != 0.0
        or result.white != 0.0
    )


def test_component_functions_return_values():
    board = cshogi.Board()

    defense = defense_thinness_score(
        board,
        cshogi.BLACK,
    )

    attack = attack_concentration_score(
        board,
        cshogi.BLACK,
    )

    invasion = invasion_risk_score(
        board,
        cshogi.BLACK,
    )

    assert isinstance(
        defense,
        float,
    )

    assert isinstance(
        attack,
        float,
    )

    assert isinstance(
        invasion,
        float,
    )

    assert 0.0 <= defense <= 1.0
    assert 0.0 <= attack <= 1.0
    assert 0.0 <= invasion <= 1.0