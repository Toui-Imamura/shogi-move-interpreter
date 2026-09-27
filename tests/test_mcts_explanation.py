import cshogi

from interpreter.feature_pipeline import (
    compute_feature_pipeline,
)
from interpreter.mcts_explanation import (
    describe_move,
    extract_mcts_changes,
)


def test_describe_normal_move():
    board = cshogi.Board()

    move = board.move_from_usi("7g7f")

    description = describe_move(
        board,
        move,
    )

    assert description.move_usi == "7g7f"
    assert description.piece_name == "歩"
    assert description.from_square == "7g"
    assert description.to_square == "7f"
    assert description.is_drop is False


def test_extract_mcts_changes_without_mcts():
    board = cshogi.Board()

    move = board.move_from_usi("7g7f")

    result = compute_feature_pipeline(
        board,
        move,
    )

    changes = extract_mcts_changes(
        result,
    )

    assert changes == []


def test_extract_mcts_changes_returns_sorted_changes():
    class FakeF38:
        weighted_changes = {
            "F01": 0.01,
            "F07": 0.40,
            "F11": -0.80,
            "F24": 0.20,
        }

    class FakeMCTS:
        f38 = FakeF38()

    class FakePipeline:
        mcts = FakeMCTS()

    changes = extract_mcts_changes(
        FakePipeline(),
        threshold=0.05,
        max_features=3,
    )

    assert [item.feature_name for item in changes] == [
        "F11",
        "F07",
        "F24",
    ]

    assert changes[0].change == -0.80
