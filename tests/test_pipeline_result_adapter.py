import cshogi
import pytest

from interpreter.feature_pipeline import (
    compute_feature_pipeline,
)

from interpreter.pipeline_result_adapter import (
    pipeline_result_to_dict,
    pipeline_result_to_vector,
)


def make_pipeline_result():
    board = cshogi.Board()

    move = board.move_from_usi("7g7f")

    return compute_feature_pipeline(
        before=board,
        move=move,
        variation_moves=[
            "7g7f",
            "3c3d",
            "2g2f",
        ],
    )


def test_pipeline_result_to_dict_has_40_features():
    result = make_pipeline_result()

    values = pipeline_result_to_dict(result)

    assert len(values) == 40

    for index in range(1, 41):
        assert f"F{index:02d}" in values


def test_pipeline_result_to_vector_has_length_40():
    result = make_pipeline_result()

    vector = pipeline_result_to_vector(result)

    assert len(vector) == 40
    assert all(
        isinstance(value, float)
        for value in vector
    )


def test_pipeline_result_f40_matches_pipeline_result():
    result = make_pipeline_result()

    values = pipeline_result_to_dict(result)

    assert values["F40"] == pytest.approx(
        result.f40_degree
    )


def test_pipeline_result_without_variation():
    board = cshogi.Board()
    move = board.move_from_usi("7g7f")

    result = compute_feature_pipeline(
        before=board,
        move=move,
    )

    values = pipeline_result_to_dict(result)

    assert len(values) == 40
    assert values["F34"] == 0.0
    assert values["F35"] == 0.0
    assert values["F36"] == 0.0
    assert values["F37"] == 0.0
    assert values["F38"] == 0.0
    assert values["F39"] == 0.0


def test_pipeline_result_with_mcts():
    board = cshogi.Board()
    move = board.move_from_usi("7g7f")

    variation1 = compute_feature_pipeline(
        before=board,
        move=move,
        variation_moves=[
            "7g7f",
            "3c3d",
        ],
    ).variation

    variation2 = compute_feature_pipeline(
        before=board,
        move=move,
        variation_moves=[
            "7g7f",
            "8c8d",
        ],
    ).variation

    result = compute_feature_pipeline(
        before=board,
        move=move,
        mcts_variations=[
            {
                "result": variation1,
                "visits": 10,
            },
            {
                "result": variation2,
                "visits": 5,
            },
        ],
    )

    values = pipeline_result_to_dict(result)

    assert values["F37"] >= 0.0
    assert values["F38"] >= 0.0
    assert 0.0 <= values["F39"] <= 1.0


def test_pipeline_result_to_dict_includes_variation_f13_and_f15():
    """Variationで計算されたF13・F15を辞書へ反映する。"""
    board = cshogi.Board()
    move = board.move_from_usi("7g7f")

    result = compute_feature_pipeline(
        before=board,
        move=move,
        variation_moves=[
            "7g7f",
            "3c3d",
            "2g2f",
        ],
    )

    values = pipeline_result_to_dict(result)

    assert len(values) == 40
    assert values["F13"] == pytest.approx(
        result.variation.variation_deltas["F13"]
    )
    assert values["F15"] == pytest.approx(
        result.variation.variation_deltas["F15"]
    )


def test_pipeline_result_to_vector_preserves_variation_f15():
    """固定長ベクトルでもF15の値を保持する。"""
    board = cshogi.Board()
    move = board.move_from_usi("7g7f")

    result = compute_feature_pipeline(
        before=board,
        move=move,
        variation_moves=[
            "7g7f",
            "3c3d",
            "2g2f",
        ],
    )

    vector = pipeline_result_to_vector(result)

    assert len(vector) == 40
    assert vector[14] == pytest.approx(
        result.variation.variation_deltas["F15"]
    )
