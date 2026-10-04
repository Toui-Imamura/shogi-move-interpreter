from __future__ import annotations

import math

import cshogi
import torch

from training.nnue.dataset import (
    NNUEPositionDataset,
    load_position_records,
    teacher_score_to_target,
)


DATASET_PATH = (
    "data/processed/"
    "teacher_values_wcsc36_295games.jsonl"
)


def test_load_position_records() -> None:
    records = load_position_records(DATASET_PATH)

    assert len(records) > 0

    first = records[0]

    assert first.ply >= 1
    assert first.sfen
    assert isinstance(
        first.teacher_score_cp_black,
        float,
    )
    assert first.side_to_move in (
        cshogi.BLACK,
        cshogi.WHITE,
    )


def test_teacher_score_to_target() -> None:
    assert teacher_score_to_target(0.0) == 0.0

    positive = teacher_score_to_target(1000.0)
    negative = teacher_score_to_target(-1000.0)

    assert positive > 0.0
    assert negative < 0.0

    assert math.isclose(
        positive,
        math.tanh(1.0),
        rel_tol=1e-7,
        abs_tol=1e-7,
    )

    assert math.isclose(
        negative,
        math.tanh(-1.0),
        rel_tol=1e-7,
        abs_tol=1e-7,
    )


def test_dataset() -> None:
    dataset = NNUEPositionDataset(DATASET_PATH)

    assert len(dataset) > 0

    sample = dataset[0]

    assert "sfen" in sample
    assert "ply" in sample
    assert "side_to_move" in sample
    assert "teacher_score_cp_black" in sample
    assert "target" in sample

    assert isinstance(
        sample["sfen"],
        str,
    )

    assert isinstance(
        sample["ply"],
        int,
    )

    assert sample["side_to_move"] in (
        cshogi.BLACK,
        cshogi.WHITE,
    )

    assert isinstance(
        sample["teacher_score_cp_black"],
        float,
    )

    assert isinstance(
        sample["target"],
        torch.Tensor,
    )

    assert sample["target"].dtype.is_floating_point
