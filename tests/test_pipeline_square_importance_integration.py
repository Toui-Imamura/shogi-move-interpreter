import cshogi
import pytest

from features.square_importance import (
    changed_square_effects,
)
from interpreter.feature_pipeline import compute_feature_pipeline
from interpreter.mcts_square_importance import (
    analyze_pipeline_square_importance,
)
from interpreter.variation_features import compute_variation_features
from scripts.run_kifu_explanation import serialize_square_analysis

def test_pipeline_square_importance_with_real_mcts_variations():
    """実際の特徴量パイプラインからマス重要度を解析できる。"""
    before = cshogi.Board()
    move = before.move_from_usi("7g7f")

    # パイプラインのMCTSは対象手を指した後の局面から開始する。
    after = before.copy()
    after.push(move)

    variation_moves_1 = ["3c3d", "2g2f"]
    variation_moves_2 = ["8c8d", "2g2f"]

    variations = [
        {
            "visits": 10,
            "result": compute_variation_features(
                after, variation_moves_1
            ),
        },
        {
            "visits": 5,
            "result": compute_variation_features(
                after, variation_moves_2
            ),
        },
    ]

    pipeline_result = compute_feature_pipeline(
        before=before,
        move=move,
        mcts_variations=variations,
    )

    assert pipeline_result.mcts is not None
    assert len(pipeline_result.mcts.variation_results) == 2
    assert list(pipeline_result.mcts.variation_visits) == [10, 5]

    result = analyze_pipeline_square_importance(pipeline_result)

    assert len(result.aggregated) > 0
    assert len(result.variations) > 0
    assert all(0 <= item.square < 81 for item in result.aggregated)
    assert all(item.total_visits >= 0 for item in result.aggregated)


def test_pipeline_square_importance_without_mcts_is_rejected():
    """MCTSを実行していないパイプライン結果は解析できない。"""
    before = cshogi.Board()
    move = before.move_from_usi("7g7f")

    pipeline_result = compute_feature_pipeline(
        before=before,
        move=move,
    )

    with pytest.raises(ValueError):
        analyze_pipeline_square_importance(pipeline_result)

def test_same_position_has_no_immediate_square_changes():
    """同一局面では即時のマス変化が記録されない。"""
    board = cshogi.Board()

    result = serialize_square_analysis(
        before=board,
        after=board.copy(),
        mover_color=board.turn,
    )

    assert result["immediate_changes"] == []


def test_pawn_move_marks_source_and_destination():
    """歩の移動元と移動先が駒配置の変化として記録される。"""
    before = cshogi.Board()
    after = before.copy()
    after.push_usi("7g7f")

    result = serialize_square_analysis(
        before=before,
        after=after,
        mover_color=before.turn,
    )

    changes = {item["usi"]: item for item in result["immediate_changes"]}

    assert "piece_placement" in changes["7g"]["change_types"]
    assert "piece_placement" in changes["7f"]["change_types"]


def test_change_types_match_underlying_square_effects():
    """分類ラベルが実際のマス差分と一致する。"""
    before = cshogi.Board()
    after = before.copy()
    after.push_usi("7g7f")

    result = serialize_square_analysis(
        before=before,
        after=after,
        mover_color=before.turn,
    )

    expected = {
        new.square: (old, new)
        for old, new in changed_square_effects(before, after)
    }

    for item in result["immediate_changes"]:
        old, new = expected[item["square"]]
        types = set(item["change_types"])

        assert ("piece_placement" in types) == (old.piece != new.piece)
        assert ("control_change" in types) == (
            old.black_attackers != new.black_attackers
            or old.white_attackers != new.white_attackers
        )
        assert ("contested_change" in types) == (
            old.contested != new.contested
        )
        assert ("importance_change" in types) == (
            old.importance != new.importance
        )
