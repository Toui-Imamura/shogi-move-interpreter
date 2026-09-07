import json

import cshogi

from training.transition_dataset import (
    build_transition,
    convert_jsonl_to_transition_jsonl,
)


INITIAL_SFEN = (
    "lnsgkgsnl/1r5b1/"
    "ppppppppp/9/9/9/"
    "PPPPPPPPP/1B5R1/"
    "LNSGKGSNL b - 1"
)


def test_build_transition():
    """実際の指し手による特徴量変化を計算できる。"""

    record = {
        "game_id": 0,
        "ply": 1,
        "sfen": INITIAL_SFEN,
        "move": "7g7f",
        "result": 2,
    }

    converted = build_transition(record)

    assert converted["game_id"] == 0
    assert converted["ply"] == 1
    assert converted["move"] == "7g7f"
    assert converted["result"] == 2

    deltas = converted["feature_deltas"]

    assert isinstance(deltas, dict)
    assert len(deltas) > 0

    assert "F01" in deltas
    assert "F06" in deltas
    assert "F07" in deltas

    # 7g7fによって実際に変化が発生する
    assert deltas["F06"] != 0.0
    assert deltas["F07"] != 0.0


def test_transition_matches_board_state():
    """保存された指し手が合法手として適用できる。"""

    record = {
        "game_id": 0,
        "ply": 1,
        "sfen": INITIAL_SFEN,
        "move": "7g7f",
        "result": 2,
    }

    board = cshogi.Board(
        sfen=record["sfen"]
    )

    move = board.move_from_usi(
        record["move"]
    )

    assert board.is_legal(move)

    after = board.copy()
    after.push(move)

    assert after.sfen() != board.sfen()


def test_convert_jsonl_to_transition_jsonl(
    tmp_path,
):
    """JSONL全体を変換できる。"""

    input_path = (
        tmp_path / "positions.jsonl"
    )

    output_path = (
        tmp_path / "transition_positions.jsonl"
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
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    count = convert_jsonl_to_transition_jsonl(
        input_path,
        output_path,
    )

    assert count == 2
    assert output_path.exists()

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        converted_records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    assert len(converted_records) == 2

    assert (
        converted_records[0]["move"]
        == "7g7f"
    )

    assert (
        "feature_deltas"
        in converted_records[0]
    )

    assert (
        converted_records[0]["feature_deltas"]
        ["F06"]
        != 0.0
    )


def test_illegal_move_is_rejected():
    """不正な指し手を拒否する。"""

    record = {
        "game_id": 0,
        "ply": 1,
        "sfen": INITIAL_SFEN,
        "move": "1a1b",
        "result": 2,
    }

    try:
        build_transition(record)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "illegal move should raise ValueError"
        )
