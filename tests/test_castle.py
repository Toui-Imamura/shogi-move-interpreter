import cshogi

from features.castle import (
    f29_castle_progress,
    f29_castle_progress_change,
    king_position_score,
    gold_silver_placement_score,
    defense_density_score,
)
from features.transition import apply_usi_move


def test_initial_castle_progress_is_valid():
    board = cshogi.Board()

    black = f29_castle_progress(
        board,
        cshogi.BLACK,
    )

    white = f29_castle_progress(
        board,
        cshogi.WHITE,
    )

    assert 0.0 <= black.score <= 1.0
    assert 0.0 <= white.score <= 1.0

    assert 0.0 <= black.king_position <= 1.0
    assert 0.0 <= black.gold_silver <= 1.0
    assert 0.0 <= black.defense_density <= 1.0


def test_initial_black_white_are_symmetric():
    board = cshogi.Board()

    black = f29_castle_progress(
        board,
        cshogi.BLACK,
    )

    white = f29_castle_progress(
        board,
        cshogi.WHITE,
    )

    assert abs(
        black.score - white.score
    ) < 1e-9

    assert abs(
        black.king_position - white.king_position
    ) < 1e-9

    assert abs(
        black.gold_silver - white.gold_silver
    ) < 1e-9

    assert abs(
        black.defense_density - white.defense_density
    ) < 1e-9


def test_components_are_non_negative():
    board = cshogi.Board()

    black = f29_castle_progress(
        board,
        cshogi.BLACK,
    )

    assert black.king_position >= 0.0
    assert black.gold_silver >= 0.0
    assert black.defense_density >= 0.0


def test_same_position_change_is_zero():
    board = cshogi.Board()

    result = f29_castle_progress_change(
        board,
        board,
    )

    assert abs(result.black) < 1e-9
    assert abs(result.white) < 1e-9
    assert abs(result.difference) < 1e-9


def test_king_move_changes_castle_progress():
    board = cshogi.Board()

    before = board.copy()

    board = apply_usi_move(
        board,
        "7g7f",
    )

    board = apply_usi_move(
        board,
        "3c3d",
    )

    board = apply_usi_move(
        board,
        "2g2f",
    )

    board = apply_usi_move(
        board,
        "8c8d",
    )

    board = apply_usi_move(
        board,
        "6i7h",
    )

    result = f29_castle_progress_change(
        before,
        board,
    )

    assert result.black != 0.0


def test_king_position_changes_when_king_moves():
    board = cshogi.Board()

    before = f29_castle_progress(
        board,
        cshogi.BLACK,
    )

    board = apply_usi_move(
        board,
        "5i6h",
    )

    after = f29_castle_progress(
        board,
        cshogi.BLACK,
    )

    assert after.king_position > before.king_position


def test_component_functions_return_normalized_values():
    board = cshogi.Board()

    king = king_position_score(
        board,
        cshogi.BLACK,
    )

    gold_silver = gold_silver_placement_score(
        board,
        cshogi.BLACK,
    )

    density = defense_density_score(
        board,
        cshogi.BLACK,
    )

    assert 0.0 <= king <= 1.0
    assert 0.0 <= gold_silver <= 1.0
    assert 0.0 <= density <= 1.0