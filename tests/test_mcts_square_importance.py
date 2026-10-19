import cshogi
import pytest

from interpreter.mcts_square_importance import (
    aggregate_square_importance,
    analyze_mcts_square_importance,
    analyze_pipeline_square_importance,
    compare_square_importance,
)


def test_same_position_has_no_square_importance_changes():
    board = cshogi.Board()

    result = compare_square_importance(board, board)

    assert result.changes == ()


def test_move_detects_square_importance_changes():
    before = cshogi.Board()
    after = cshogi.Board()
    after.push_usi("7g7f")

    result = compare_square_importance(before, after)

    assert result.changes
    assert all(0 <= item.square < 81 for item in result.changes)


def test_change_has_correct_direction():
    before = cshogi.Board()
    after = cshogi.Board()
    after.push_usi("7g7f")

    result = compare_square_importance(before, after)

    for change in result.changes:
        if change.delta > 0:
            assert change.direction == "increase"
        elif change.delta < 0:
            assert change.direction == "decrease"
        else:
            assert change.direction == "unchanged"


def test_negative_visits_are_rejected():
    board = cshogi.Board()

    with pytest.raises(ValueError):
        compare_square_importance(board, board, visits=-1)


def test_aggregate_combines_repeated_square_changes():
    before = cshogi.Board()
    after = cshogi.Board()
    after.push_usi("7g7f")

    first = compare_square_importance(
        before, after, variation_index=0, visits=3
    )
    second = compare_square_importance(
        before, after, variation_index=1, visits=5
    )

    result = aggregate_square_importance([first, second])

    assert result
    assert all(item.variation_count >= 1 for item in result)
    assert all(item.total_visits >= 0 for item in result)


def test_mcts_analysis_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        analyze_mcts_square_importance(
            [[cshogi.Board()]],
            [],
        )


def test_mcts_analysis_has_one_record_per_transition():
    start = cshogi.Board()
    after_one = start.copy()
    after_one.push_usi("7g7f")
    after_two = after_one.copy()
    after_two.push_usi("3c3d")

    result = analyze_mcts_square_importance(
        [[start, after_one, after_two]],
        [10],
    )

    assert len(result.variations) == 2
    assert result.aggregated
    assert all(item.variation_count >= 1 for item in result.aggregated)


from types import SimpleNamespace


def test_pipeline_square_importance_uses_mcts_results():
    start = cshogi.Board()

    after_one = start.copy()
    after_one.push_usi("7g7f")

    after_two = after_one.copy()
    after_two.push_usi("3c3d")

    variation_result = SimpleNamespace(
        positions=[start, after_one, after_two]
    )
    pipeline_result = SimpleNamespace(
        mcts=SimpleNamespace(
            variation_results=[variation_result],
            variation_visits=[10],
        )
    )

    result = analyze_pipeline_square_importance(pipeline_result)

    assert len(result.variations) == 2
    assert result.aggregated


def test_pipeline_square_importance_rejects_missing_mcts():
    pipeline_result = SimpleNamespace(mcts=None)

    with pytest.raises(ValueError):
        analyze_pipeline_square_importance(pipeline_result)


def test_pipeline_square_importance_rejects_mismatched_lengths():
    pipeline_result = SimpleNamespace(
        mcts=SimpleNamespace(
            variation_results=[],
            variation_visits=[10],
        )
    )

    with pytest.raises(ValueError):
        analyze_pipeline_square_importance(pipeline_result)


def test_zero_visit_variation_is_excluded_from_aggregation():
    before = cshogi.Board()
    after = cshogi.Board()
    after.push_usi("7g7f")

    visited = compare_square_importance(
        before, after, variation_index=0, visits=5
    )
    unvisited = compare_square_importance(
        before, after, variation_index=1, visits=0
    )

    actual = aggregate_square_importance([visited, unvisited])
    expected = aggregate_square_importance([visited])

    assert actual == expected


def test_mcts_analysis_skips_zero_visit_variation():
    before = cshogi.Board()
    after = before.copy()
    after.push_usi("7g7f")

    result = analyze_mcts_square_importance(
        [[before, after], [before, after]],
        [5, 0],
    )

    assert len(result.variations) == 1
    assert result.variations[0].variation_index == 0
    assert result.aggregated
    assert all(
        item.variation_count == 1
        for item in result.aggregated
    )
