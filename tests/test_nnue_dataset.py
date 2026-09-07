from __future__ import annotations

import cshogi
import pytest

from training.nnue.dataset import (
    load_position_records,
    result_to_target,
    NNUEPositionDataset,
)


def test_load_position_records():

    records = load_position_records(
        "data/processed/positions.jsonl"
    )

    assert len(records) == 10

    first = records[0]

    assert first.ply == 1
    assert first.sfen
    assert first.result == 2
    assert first.side_to_move == cshogi.BLACK


def test_result_to_target():

    assert (
        result_to_target(
            cshogi.BLACK_WIN,
            cshogi.BLACK,
        )
        == 1.0
    )

    assert (
        result_to_target(
            cshogi.BLACK_WIN,
            cshogi.WHITE,
        )
        == -1.0
    )

    assert (
        result_to_target(
            cshogi.WHITE_WIN,
            cshogi.WHITE,
        )
        == 1.0
    )

    assert (
        result_to_target(
            cshogi.WHITE_WIN,
            cshogi.BLACK,
        )
        == -1.0
    )

    assert (
        result_to_target(
            cshogi.DRAW,
            cshogi.BLACK,
        )
        == 0.0
    )


def test_dataset():

    dataset = NNUEPositionDataset(
        "data/processed/positions.jsonl"
    )

    assert len(dataset) == 10

    sample = dataset[0]

    assert "sfen" in sample
    assert "ply" in sample
    assert "side_to_move" in sample
    assert "target" in sample

    assert sample["target"].dtype.is_floating_point


def test_unknown_result():

    with pytest.raises(ValueError):

        result_to_target(
            999,
            cshogi.BLACK,
        )
