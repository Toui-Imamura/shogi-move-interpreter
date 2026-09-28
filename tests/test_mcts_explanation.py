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


def test_find_transition_move():
    import cshogi

    from interpreter.mcts_explanation import (
        find_transition_move,
    )

    before = cshogi.Board()
    move = before.move_from_usi("7g7f")

    after = before.copy()
    after.push(move)

    detected = find_transition_move(
        before,
        after,
    )

    assert detected == move


def test_extract_position_changes():
    import cshogi

    from interpreter.mcts_explanation import (
        extract_position_changes,
    )

    before = cshogi.Board()
    move = before.move_from_usi("7g7f")

    after = before.copy()
    after.push(move)

    changes = extract_position_changes(
        before,
        after,
    )

    assert len(changes) == 1

    change = changes[0]

    assert change.move_usi == "7g7f"
    assert change.piece_name == "歩"
    assert change.from_square == "7g"
    assert change.to_square == "7f"
    assert change.is_drop is False


def test_extract_variation_moves():
    import cshogi

    from interpreter.mcts_explanation import (
        extract_variation_moves,
    )

    board = cshogi.Board()

    move1 = board.move_from_usi("7g7f")
    board.push(move1)

    move2 = board.move_from_usi("3c3d")
    board.push(move2)

    positions = [
        cshogi.Board(),
    ]

    position1 = cshogi.Board()
    position1.push(
        position1.move_from_usi("7g7f")
    )

    position2 = position1.copy()
    position2.push(
        position2.move_from_usi("3c3d")
    )

    positions.extend(
        [
            position1,
            position2,
        ]
    )

    changes = extract_variation_moves(
        positions,
    )

    assert len(changes) == 2

    assert changes[0].move_usi == "7g7f"
    assert changes[0].piece_name == "歩"

    assert changes[1].move_usi == "3c3d"
    assert changes[1].piece_name == "歩"


def test_describe_mcts_variations():
    """MCTS variationを具体的な指し手列へ変換できることを確認する。"""

    import cshogi

    from interpreter.feature_pipeline import compute_pipeline
    from interpreter.mcts_explanation import (
        describe_mcts_variations,
    )
    from interpreter.variation_features import (
        compute_variation_features,
    )

    board = cshogi.Board()

    variation_result = compute_variation_features(
        board,
        ["7g7f", "3c3d", "2g2f"],
    )

    pipeline_result = compute_pipeline(
        before=board,
        move=board.move_from_usi("7g7f"),
        mcts_variations=[
            {
                "result": variation_result,
                "visits": 5,
            }
        ],
    )

    descriptions = describe_mcts_variations(
        pipeline_result
    )

    assert len(descriptions) == 1

    description = descriptions[0]

    assert description.variation_index == 0
    assert description.visits == 5

    assert len(description.moves) == 3

    assert description.moves[0].move_usi == "7g7f"
    assert description.moves[1].move_usi == "3c3d"
    assert description.moves[2].move_usi == "2g2f"


def test_extract_feature_transition_evidence():
    """特徴量の変化と具体的な指し手を対応付けられることを確認する。"""

    import cshogi

    from interpreter.feature_pipeline import compute_pipeline
    from interpreter.mcts_explanation import (
        extract_feature_transition_evidence,
    )
    from interpreter.variation_features import (
        compute_variation_features,
    )

    board = cshogi.Board()

    variation_result = compute_variation_features(
        board,
        ["7g7f", "3c3d", "2g2f"],
    )

    pipeline_result = compute_pipeline(
        before=board,
        move=board.move_from_usi("7g7f"),
        mcts_variations=[
            {
                "result": variation_result,
                "visits": 5,
            }
        ],
    )

    evidence = extract_feature_transition_evidence(
        pipeline_result,
        feature_name="F07",
        threshold=0.0,
    )

    assert isinstance(evidence, list)

    for item in evidence:
        assert item.feature_name == "F07"
        assert item.variation_index == 0
        assert item.move_index >= 0
        assert isinstance(item.move.move_usi, str)
        assert isinstance(item.change, float)
