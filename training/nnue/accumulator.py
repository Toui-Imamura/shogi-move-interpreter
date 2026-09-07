from __future__ import annotations

from dataclasses import dataclass

import torch
import cshogi

from training.nnue.features import (
    NNUEFeature,
    extract_features,
    feature_to_id,
)


BLACK = cshogi.BLACK
WHITE = cshogi.WHITE


def features_to_ids(
    features: list[NNUEFeature],
) -> list[int]:
    """
    NNUEFeatureのリストをEmbedding用IDへ変換する。
    """

    return [
        feature_to_id(feature)
        for feature in features
    ]


def accumulate(
    embedding: torch.nn.Embedding,
    feature_ids: list[int],
) -> torch.Tensor:
    """
    特徴IDに対応するEmbeddingを全て加算して
    Accumulatorを生成する。
    """

    if not feature_ids:
        return torch.zeros(
            embedding.embedding_dim,
            device=embedding.weight.device,
            dtype=embedding.weight.dtype,
        )

    ids = torch.tensor(
        feature_ids,
        dtype=torch.long,
        device=embedding.weight.device,
    )

    return embedding(ids).sum(dim=0)


def board_to_accumulators(
    board: cshogi.Board,
    embedding: torch.nn.Embedding,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    cshogi.BoardからBlack/WhiteのAccumulatorを生成する。
    """

    black_features = extract_features(
        board,
        BLACK,
    )

    white_features = extract_features(
        board,
        WHITE,
    )

    black_ids = features_to_ids(
        black_features,
    )

    white_ids = features_to_ids(
        white_features,
    )

    accumulator_black = accumulate(
        embedding,
        black_ids,
    )

    accumulator_white = accumulate(
        embedding,
        white_ids,
    )

    return (
        accumulator_black,
        accumulator_white,
    )


@dataclass
class NNUEAccumulator:
    """
    NNUEのBlack/White Accumulatorを保持する。

    current_board:
        Accumulatorが対応している局面。

    black:
        Black視点Accumulator

    white:
        White視点Accumulator
    """

    current_board: cshogi.Board
    black: torch.Tensor
    white: torch.Tensor


def create_accumulator(
    board: cshogi.Board,
    embedding: torch.nn.Embedding,
) -> NNUEAccumulator:
    """
    局面からNNUEAccumulatorを生成する。
    """

    black, white = board_to_accumulators(
        board,
        embedding,
    )

    return NNUEAccumulator(
        current_board=board.copy(),
        black=black,
        white=white,
    )


def update_accumulator(
    accumulator: NNUEAccumulator,
    board_after: cshogi.Board,
    embedding: torch.nn.Embedding,
) -> NNUEAccumulator:
    """
    指し手後の局面に対してAccumulatorを更新する。

    現段階では正しさを優先し、

        全特徴を再計算したAccumulator

    と

        差分更新Accumulator

    の一致を確認するための実装とする。

    本格的な高速差分更新は、
    特徴表現を固定した後に実装する。
    """

    black_after, white_after = board_to_accumulators(
        board_after,
        embedding,
    )

    return NNUEAccumulator(
        current_board=board_after.copy(),
        black=black_after,
        white=white_after,
    )


def evaluate_accumulator(
    accumulator: NNUEAccumulator,
    model: torch.nn.Module,
) -> torch.Tensor:
    """
    AccumulatorをNNUEへ入力して評価値を取得する。
    """

    return model(
        accumulator.black.unsqueeze(0),
        accumulator.white.unsqueeze(0),
    ).squeeze(0)


def evaluate_board(
    board: cshogi.Board,
    model: torch.nn.Module,
) -> float:
    """
    盤面を直接NNUEで評価する。
    """

    black, white = board_to_accumulators(
        board,
        model.embedding,
    )

    output = model(
        black.unsqueeze(0),
        white.unsqueeze(0),
    )

    return float(output.item())
