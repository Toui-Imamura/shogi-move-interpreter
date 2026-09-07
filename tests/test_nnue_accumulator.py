from __future__ import annotations

import cshogi
import torch

from training.nnue.features import (
    num_feature_ids,
)
from training.nnue.network import NNUE
from training.nnue.accumulator import (
    accumulate,
    board_to_accumulators,
    features_to_ids,
)


def test_features_to_ids():

    board = cshogi.Board()

    black_features = (
        __import__(
            "training.nnue.features",
            fromlist=["extract_features"],
        )
        .extract_features(
            board,
            cshogi.BLACK,
        )
    )

    ids = features_to_ids(
        black_features,
    )

    assert len(ids) > 0

    assert all(
        0 <= feature_id < num_feature_ids()
        for feature_id in ids
    )


def test_accumulate():

    model = NNUE(
        num_features=num_feature_ids(),
        accumulator_size=256,
        hidden_size=32,
    )

    feature_ids = [1, 2, 3]

    accumulator = accumulate(
        model.embedding,
        feature_ids,
    )

    assert accumulator.shape == (256,)


def test_board_to_accumulators():

    board = cshogi.Board()

    model = NNUE(
        num_features=num_feature_ids(),
        accumulator_size=256,
        hidden_size=32,
    )

    black, white = board_to_accumulators(
        board,
        model.embedding,
    )

    assert black.shape == (256,)
    assert white.shape == (256,)


def test_nnue_from_board():

    board = cshogi.Board()

    model = NNUE(
        num_features=num_feature_ids(),
        accumulator_size=256,
        hidden_size=32,
    )

    black, white = board_to_accumulators(
        board,
        model.embedding,
    )

    output = model(
        black.unsqueeze(0),
        white.unsqueeze(0),
    )

    assert output.shape == (1,)
    assert torch.isfinite(output).all()


def test_create_and_update_accumulator():

    import torch

    from training.nnue.accumulator import (
        create_accumulator,
        update_accumulator,
    )

    board = cshogi.Board()

    model = NNUE(
        num_features=num_feature_ids(),
        accumulator_size=256,
        hidden_size=32,
    )

    accumulator_before = create_accumulator(
        board,
        model.embedding,
    )

    move = board.move_from_usi("7g7f")

    assert board.is_legal(move)

    board_after = board.copy()
    board_after.push(move)

    accumulator_after = update_accumulator(
        accumulator_before,
        board_after,
        model.embedding,
    )

    assert accumulator_after.black.shape == (256,)
    assert accumulator_after.white.shape == (256,)

    assert torch.isfinite(
        accumulator_after.black
    ).all()

    assert torch.isfinite(
        accumulator_after.white
    ).all()
