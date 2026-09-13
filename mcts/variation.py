from __future__ import annotations

from dataclasses import dataclass

import cshogi

from interpreter.variation_features import (
    VariationFeatureResult,
    compute_variation_features,
)
from mcts.search import MCTSVariation


@dataclass
class EvaluatedVariation:
    """
    MCTSで得られた変化手順と、
    特徴量計算結果をまとめたデータ。
    """

    result: VariationFeatureResult
    visits: int
    value: float
    moves: list[int]


def moves_to_usi(
    moves: list[int],
) -> list[str]:
    """
    cshogiの指し手ID列をUSI文字列列へ変換する。
    """

    return [
        cshogi.move_to_usi(move)
        for move in moves
    ]


def convert_variation(
    initial_board: cshogi.Board,
    variation: MCTSVariation,
) -> EvaluatedVariation:
    """
    MCTSVariationをVariationFeatureResultへ変換する。
    """

    if initial_board is None:
        raise ValueError(
            "initial_board must not be None"
        )

    if variation is None:
        raise ValueError(
            "variation must not be None"
        )

    usi_moves = moves_to_usi(
        variation.moves
    )

    result = compute_variation_features(
        initial_board,
        usi_moves,
    )

    return EvaluatedVariation(
        result=result,
        visits=int(variation.visits),
        value=float(variation.value),
        moves=list(variation.moves),
    )


def convert_variations(
    initial_board: cshogi.Board,
    variations: list[MCTSVariation],
) -> list[EvaluatedVariation]:
    """
    複数のMCTS変化手順を特徴量計算結果へ変換する。
    """

    if initial_board is None:
        raise ValueError(
            "initial_board must not be None"
        )

    if variations is None:
        raise ValueError(
            "variations must not be None"
        )

    return [
        convert_variation(
            initial_board,
            variation,
        )
        for variation in variations
    ]


def to_mcts_feature_inputs(
    variations: list[EvaluatedVariation],
) -> list[dict]:
    """
    compute_mcts_features()へ渡す入力形式へ変換する。
    """

    if variations is None:
        raise ValueError(
            "variations must not be None"
        )

    return [
        {
            "result": variation.result,
            "visits": variation.visits,
        }
        for variation in variations
    ]
