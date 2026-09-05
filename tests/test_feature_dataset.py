"""
tests/test_feature_dataset.py
"""

import json

import cshogi

from training.feature_dataset import (
    convert_jsonl_to_feature_jsonl,
    convert_position_record,
    extract_position_features,
)


INITIAL_SFEN = (
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/"
    "PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
)


def test_extract_position_features():
    """初期局面から特徴量を抽出できる。"""

    board = cshogi.Board(
        sfen=INITIAL_SFEN
    )

    features = extract_position_features(
        board
    )

    assert isinstance(features, dict)

    assert "F01" in features
    assert "F02" in features
    assert "F04" in features
    assert "F05" in features
    assert "F06" in features
    assert "F33" in features

    for value in features.values():
        assert isinstance(value, float)


def test_feature_keys_are_sorted():
    """特徴量キーがF番号順になっている。"""

    board = cshogi.Board(
        sfen=INITIAL_SFEN
    )

    features = extract_position_features(
        board
    )

    keys = list(features.keys())

    assert keys == sorted(keys)


def test_convert_position_record():
    """局面レコードに特徴量を追加できる。"""

    record = {
        "game_id": 0,
        "ply": 1,
        "sfen": INITIAL_SFEN,
        "move": "7g7f",
        "result": 2,
    }

    converted = convert_position_record(
        record
    )

    assert converted["game_id"] == 0
    assert converted["ply"] == 1
    assert converted["move"] == "7g7f"

    assert "features" in converted
    assert isinstance(
        converted["features"],
        dict,
    )


def test_convert_jsonl_to_feature_jsonl(
    tmp_path,
):
    """JSONL全体を変換できる。"""

    input_path = (
        tmp_path / "positions.jsonl"
    )

    output_path = (
        tmp_path / "feature_positions.jsonl"
    )

    records = [
        {
            "game_id": 0,
            "ply": 1,
            "sfen": INITIAL_SFEN,
            "move": "7g7f",
            "result": 2,
        },
        {
            "game_id": 0,
            "ply": 2,
            "sfen": (
                "lnsgkgsnl/1r5b1/"
                "ppppppppp/9/9/2P6/"
                "PP1PPPPPP/1B5R1/"
                "LNSGKGSNL w - 2"
            ),
            "move": "3c3d",
            "result": 2,
        },
    ]

    with input_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(record)
                + "\n"
            )

    count = convert_jsonl_to_feature_jsonl(
        input_path,
        output_path,
    )

    assert count == 2
    assert output_path.exists()

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        output_records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    assert len(output_records) == 2
    assert "features" in output_records[0]
    assert "F01" in output_records[0]["features"]