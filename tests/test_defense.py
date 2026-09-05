import cshogi

from features.activity import BLACK, WHITE
from features.defense import (
    f23_defense,
    f23_defense_change,
    response_components,
)


def test_f23_initial_position():
    """
    初期局面でF23が計算できる。
    """

    board = cshogi.Board()

    black = f23_defense(
        board,
        BLACK,
    )

    white = f23_defense(
        board,
        WHITE,
    )

    assert black.defenders >= 0
    assert white.defenders >= 0

    assert black.king_control >= 0
    assert white.king_control >= 0

    assert 0.0 <= black.score <= 1.0
    assert 0.0 <= white.score <= 1.0


def test_f23_unchanged_position():
    """
    同一局面を比較した場合、
    防御力の変化は0になる。
    """

    board = cshogi.Board()

    result = f23_defense_change(
        board,
        board.copy(),
        BLACK,
    )

    assert result.defenders == 0
    assert result.king_control == 0
    assert result.response == 0
    assert result.score == 0


def test_f23_after_move():
    """
    指し手後にF23の変化を計算できる。
    """

    board = cshogi.Board()

    move = board.move_from_usi("7g7f")

    assert board.is_legal(move)

    after = board.copy()
    after.push(move)

    result = f23_defense_change(
        board,
        after,
        BLACK,
    )

    assert isinstance(result.defenders, float)
    assert isinstance(result.king_control, float)
    assert isinstance(result.response, float)
    assert isinstance(result.score, float)


def test_response_components():
    """
    Responseの3要素が取得できる。
    """

    board = cshogi.Board()

    result = response_components(
        board,
        BLACK,
    )

    assert result.legal_defense >= 0
    assert result.king_escape >= 0
    assert result.counter_attack >= 0