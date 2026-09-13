from __future__ import annotations

import json

import cshogi

from interpreter.dataset_io import read_jsonl, write_jsonl
from interpreter.dataset_records import (
    FeatureDatasetRecord,
    make_dataset_record,
)
from interpreter.feature_pipeline import compute_feature_pipeline


def test_make_dataset_record_has_40_features() -> None:
    board = cshogi.Board()
    move = board.move_from_usi("7g7f")

    result = compute_feature_pipeline(
        before=board,
        move=move,
    )

    record = make_dataset_record(
        game_id="test-game",
        source_file="test.csa",
        ply=1,
        side_to_move=0,
        move_usi="7g7f",
        pipeline_result=result,
    )

    assert isinstance(record, FeatureDatasetRecord)
    assert len(record.features) == 40
    assert len(record.feature_dict) == 40
    assert record.features[0] == record.feature_dict["F01"]
    assert record.features[39] == record.feature_dict["F40"]


def test_jsonl_round_trip(tmp_path) -> None:
    output_path = tmp_path / "records.jsonl"

    records = [
        {
            "game_id": "game-1",
            "ply": 1,
            "features": [0.0] * 40,
        },
        {
            "game_id": "game-2",
            "ply": 2,
            "features": [0.1] * 40,
        },
    ]

    count = write_jsonl(
        records,
        output_path,
    )

    assert count == 2
    assert output_path.exists()

    loaded = list(read_jsonl(output_path))

    assert loaded == records

    with output_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    assert len(lines) == 2

    for line in lines:
        parsed = json.loads(line)
        assert isinstance(parsed, dict)
