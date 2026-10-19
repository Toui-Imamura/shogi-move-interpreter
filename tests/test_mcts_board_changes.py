import cshogi

from interpreter.mcts_board_changes import (
    analyze_board_change,
    extract_changed_squares,
    extract_control_change,
    extract_mobility_change,
)


def test_extract_changed_squares_for_pawn_move():
    before = cshogi.Board()

    move = before.move_from_usi("7g7f")

    after = before.copy()
    after.push(move)

    changes = extract_changed_squares(
        before,
        after,
    )

    squares = {
        change.square
        for change in changes
    }

    assert "7g" in squares
    assert "7f" in squares


def test_find_pawn_move_and_piece_name():
    before = cshogi.Board()

    move = before.move_from_usi("7g7f")

    after = before.copy()
    after.push(move)

    result = analyze_board_change(
        before,
        after,
    )

    assert result.move_usi == "7g7f"
    assert result.piece_name == "歩"
    assert result.from_square == "7g"
    assert result.to_square == "7f"


def test_control_change_returns_sets():
    before = cshogi.Board()

    move = before.move_from_usi("7g7f")

    after = before.copy()
    after.push(move)

    result = extract_control_change(
        before,
        after,
        cshogi.BLACK,
    )

    assert isinstance(
        result.added_squares,
        tuple,
    )

    assert isinstance(
        result.removed_squares,
        tuple,
    )


def test_mobility_change():
    before = cshogi.Board()

    move = before.move_from_usi("7g7f")

    after = before.copy()
    after.push(move)

    result = extract_mobility_change(
        before,
        after,
        cshogi.BLACK,
    )

    assert result.after >= 0
    assert result.before >= 0
    assert result.change == (
        result.after - result.before
    )


def test_analyze_board_change_contains_feature_changes():
    before = cshogi.Board()

    move = before.move_from_usi("7g7f")

    after = before.copy()
    after.push(move)

    result = analyze_board_change(
        before,
        after,
    )

    assert len(result.changed_squares) >= 2
    assert len(result.mobility) == 2
    assert len(result.important_control) == 2
    assert len(result.defense_control) == 2


def test_extract_local_control_changes_for_pawn_move():
    import cshogi

    from interpreter.mcts_board_changes import (
        extract_local_control_changes,
    )

    before = cshogi.Board()

    move = before.move_from_usi(
        "7g7f"
    )

    after = before.copy()
    after.push(move)

    black, white = extract_local_control_changes(
        before,
        after,
    )

    assert black.color == cshogi.BLACK
    assert white.color == cshogi.WHITE

    assert isinstance(
        black.changes,
        tuple,
    )

    assert isinstance(
        white.changes,
        tuple,
    )


def test_local_control_change_contains_square_names():
    import cshogi

    from interpreter.mcts_board_changes import (
        extract_local_control_changes,
    )

    before = cshogi.Board()

    move = before.move_from_usi(
        "7g7f"
    )

    after = before.copy()
    after.push(move)

    black, white = extract_local_control_changes(
        before,
        after,
    )

    all_squares = set()

    for change in black.changes:
        all_squares.update(
            change.added_squares
        )
        all_squares.update(
            change.removed_squares
        )

    for change in white.changes:
        all_squares.update(
            change.added_squares
        )
        all_squares.update(
            change.removed_squares
        )

    for square in all_squares:
        assert len(square) == 2
        assert square[0] in "123456789"
        assert square[1] in "abcdefghi"


def test_local_control_change_does_not_report_unchanged_piece():
    import cshogi

    from interpreter.mcts_board_changes import (
        extract_local_control_changes,
    )

    before = cshogi.Board()
    after = before.copy()

    black, white = extract_local_control_changes(
        before,
        after,
    )

    assert black.changes == ()
    assert white.changes == ()


def test_analyze_variation_board_changes():
    import cshogi

    from interpreter.mcts_board_changes import (
        analyze_variation_board_changes,
    )

    board = cshogi.Board()

    position1 = board.copy()
    position1.push(
        position1.move_from_usi(
            "7g7f"
        )
    )

    position2 = position1.copy()
    position2.push(
        position2.move_from_usi(
            "3c3d"
        )
    )

    position3 = position2.copy()
    position3.push(
        position3.move_from_usi(
            "2g2f"
        )
    )

    changes = analyze_variation_board_changes(
        [
            board,
            position1,
            position2,
            position3,
        ]
    )

    assert len(changes) == 3

    assert changes[0].move_usi == "7g7f"
    assert changes[1].move_usi == "3c3d"
    assert changes[2].move_usi == "2g2f"

    assert changes[0].move_index == 0
    assert changes[1].move_index == 1
    assert changes[2].move_index == 2

    assert changes[0].black_control.changes
    assert isinstance(
        changes[0].white_control.changes,
        tuple,
    )


def test_analyze_variation_board_changes_has_bishop_effect():
    import cshogi

    from interpreter.mcts_board_changes import (
        analyze_variation_board_changes,
    )

    board = cshogi.Board()

    after = board.copy()
    after.push(
        after.move_from_usi(
            "7g7f"
        )
    )

    changes = analyze_variation_board_changes(
        [
            board,
            after,
        ]
    )

    assert len(changes) == 1

    black = changes[0].black_control

    bishop_changes = [
        change
        for change in black.changes
        if change.piece_square == "8h"
    ]

    assert bishop_changes

    bishop = bishop_changes[0]

    assert "3c" in bishop.added_squares
    assert "4d" in bishop.added_squares
    assert "5e" in bishop.added_squares
    assert "6f" in bishop.added_squares
