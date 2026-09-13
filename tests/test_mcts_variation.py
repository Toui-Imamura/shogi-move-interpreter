from __future__ import annotations

import cshogi
import torch

from interpreter.mcts_features import compute_mcts_features
from mcts.search import SimpleMCTS
from mcts.variation import (
    convert_variations,
    to_mcts_feature_inputs,
)
from training.nnue.features import num_feature_ids
from training.nnue.network import NNUE


def create_test_model() -> NNUE:
    """
    テスト用のNNUEモデルを生成する。

    学習済みモデルではなく、ランダム初期化モデルを使用する。
    このテストでは評価精度ではなく、
    MCTSから特徴量計算までの接続を確認する。
    """

    torch.manual_seed(0)

    return NNUE(
        num_features=num_feature_ids(),
        accumulator_size=256,
        hidden_size=32,
    )


def test_mcts_variations_convert_to_feature_inputs() -> None:
    """
    MCTS変化手順をVariationFeatureResultへ変換し、
    compute_mcts_features()へ渡せることを確認する。
    """

    board = cshogi.Board()
    model = create_test_model()

    searcher = SimpleMCTS(
        model=model,
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

    from interpreter.pipeline_result_adapter import (
        pipeline_result_to_dict,
        pipeline_result_to_vector,
    )
    from interpreter.feature_pipeline import (
        compute_feature_pipeline,
    )

    board = cshogi.Board()
    model = create_test_model()

    searcher = SimpleMCTS(
        model=model,
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

    feature_dict = pipeline_result_to_dict(result)
    feature_vector = pipeline_result_to_vector(result)

    assert len(feature_dict) == 40
    assert len(feature_vector) == 40

    assert list(feature_dict.keys()) == [
        f"F{i:02d}" for i in range(1, 41)
    ]

    for value in feature_vector:
        assert isinstance(value, float)

    assert "F37" in feature_dict
    assert "F38" in feature_dict
    assert "F39" in feature_dict