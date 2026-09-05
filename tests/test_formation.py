import cshogi

from features.activity import BLACK, WHITE
from features.formation import (
    F28Formation,
    F28FormationChange,
    f28_formation,
    f28_formation_change,
    mutual_control_score,
    piece_relationship_score,
    zone_connectivity_score,
)
from features.transition import apply_usi_move


def test_initial_formation():
    board = cshogi.Board()

    black = f28_formation(board, BLACK)
    white = f28_formation(board, WHITE)

    assert isinstance(black, F28Formation)
    assert isinstance(white, F28Formation)

    assert -1.0 <= black.score <= 1.0
    assert -1.0 <= white.score <= 1.0


def test_initial_formation_is_symmetric():
    board = cshogi.Board()

    black = f28_formation(board, BLACK)
    white = f28_formation(board, WHITE)

    assert black.relationship == white.relationship
    assert black.mutual_control == white.mutual_control
    assert black.zone_connectivity == white.zone_connectivity


def test_formation_components_are_non_negative():
    board = cshogi.Board()

    for color in (BLACK, WHITE):
        result = f28_formation(board, color)

        assert result.relationship >= 0.0
        assert result.mutual_control >= 0.0
        assert result.zone_connectivity >= 0.0


def test_same_position_change_is_zero():
    board = cshogi.Board()

    result = f28_formation_change(
        board,
        board,
    )

    assert isinstance(result, F28FormationChange)
    assert result.black == 0.0
    assert result.white == 0.0
    assert result.difference == 0.0


def test_formation_changes_after_move():
    board = cshogi.Board()

    after = apply_usi_move(
        board,
        "7g7f",
    )

    result = f28_formation_change(
        board,
        after,
    )

    assert isinstance(result, F28FormationChange)

    assert -1.0 <= result.black <= 1.0
    assert -1.0 <= result.white <= 1.0
    assert -2.0 <= result.difference <= 2.0


def test_component_functions():
    board = cshogi.Board()

    for color in (BLACK, WHITE):
        relationship = piece_relationship_score(
            board,
            color,
        )

        mutual_control = mutual_control_score(
            board,
            color,
        )

        zone_connectivity = zone_connectivity_score(
            board,
            color,
        )

        assert relationship >= 0.0
        assert mutual_control >= 0.0
        assert zone_connectivity >= 0.0