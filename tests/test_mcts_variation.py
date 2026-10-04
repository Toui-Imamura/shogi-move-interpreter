from __future__ import annotations

import cshogi

from interpreter.mcts_features import compute_mcts_features
from mcts.search import SimpleMCTS
from mcts.variation import (
    convert_variations,
    to_mcts_feature_inputs,
)


def create_test_evaluator():
    """
    MCTS接続テスト用の決定的な簡易評価器。

    評価精度を確認するテストではないため、
    YaneuraOuや学習済みNNUEは使用しない。
    """

    def evaluator(board: cshogi.Board) -> float:
        return float(len(list(board.legal_moves)))

    return evaluator


def test_mcts_variations_convert_to_feature_inputs() -> None:
    """
    MCTS変化手順をVariationFeatureResultへ変換し、
    compute_mcts_features()へ渡せることを確認する。
    """

    board = cshogi.Board()

    searcher = SimpleMCTS(
        evaluator=create_test_evaluator(),
        simulations=10,
        max_depth=3,
        exploration_constant=1.4,
    )

    root = searcher.search(board)

    variations = searcher.top_variations(
        root,
        num_variations=3,
        max_depth=3,
    )

    assert len(variations) > 0

    evaluated_variations = convert_variations(
        board,
        variations,
    )

    assert len(evaluated_variations) == len(
        variations
    )

    for evaluated in evaluated_variations:
        assert evaluated.result is not None
        assert evaluated.visits >= 0
        assert isinstance(evaluated.moves, list)

    feature_inputs = to_mcts_feature_inputs(
        evaluated_variations,
    )

    assert len(feature_inputs) == len(
        evaluated_variations
    )

    for item in feature_inputs:
        assert "result" in item
        assert "visits" in item
        assert item["visits"] >= 0

    mcts_result = compute_mcts_features(
        board,
        feature_inputs,
    )

    assert mcts_result is not None
    assert mcts_result.f37 is not None
    assert mcts_result.f38 is not None
    assert isinstance(mcts_result.f39, float)


def test_mcts_pipeline_produces_40_dimensional_vector() -> None:
    """
    MCTSで生成した複数変化をF37〜F39へ変換し、
    F01〜F40の40次元ベクトルへ統合できることを確認する。
    """

    from interpreter.feature_pipeline import (
        compute_feature_pipeline,
    )

    board = cshogi.Board()

    searcher = SimpleMCTS(
        evaluator=create_test_evaluator(),
        simulations=10,
        max_depth=3,
        exploration_constant=1.4,
    )

    root = searcher.search(board)

    variations = searcher.top_variations(
        root,
        num_variations=3,
        max_depth=3,
    )

    assert len(variations) > 0

    evaluated_variations = convert_variations(
        board,
        variations,
    )

    mcts_variations = to_mcts_feature_inputs(
        evaluated_variations,
    )

    move = board.move_from_usi("7g7f")

    result = compute_feature_pipeline(
        before=board,
        move=move,
        mcts_variations=mcts_variations,
    )

    assert result.mcts is not None
