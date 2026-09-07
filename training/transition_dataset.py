from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cshogi

from interpreter.feature_adapters import compute_f01_f33_deltas


def build_transition(
    record: dict[str, Any],
) -> dict[str, Any]:
    """
    1局面レコードから、実際の指し手による
    指し手前後の特徴量変化を生成する。

    入力:
        {
            "game_id": ...,
            "ply": ...,
            "sfen": ...,
            "move": "7g7f",
            "result": ...
        }

    出力:
        {
            "game_id": ...,
            "ply": ...,
            "sfen": ...,
            "move": ...,
            "result": ...,
            "feature_deltas": {...}
        }
    """

    if "sfen" not in record:
        raise ValueError("record must contain 'sfen'")

    if "move" not in record:
        raise ValueError("record must contain 'move'")

    before = cshogi.Board(
        sfen=record["sfen"]
    )

    move = before.move_from_usi(
        record["move"]
    )

    if not before.is_legal(move):
        raise ValueError(
            f"illegal move: {record['move']}"
        )

    after = before.copy()
    after.push(move)

    feature_deltas = compute_f01_f33_deltas(
        before,
        after,
        move=move,
    )

    return {
        "game_id": record.get("game_id"),
        "ply": record.get("ply"),
        "sfen": record["sfen"],
        "move": record["move"],
        "result": record.get("result"),
        "feature_deltas": feature_deltas,
    }


def convert_jsonl_to_transition_jsonl(
    input_path: str | Path,
    output_path: str | Path,
) -> int:
    """
    positions.jsonlを読み込み、
    指し手前後の特徴量変化をJSONLとして保存する。

    Returns:
        変換した局面数
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"input file not found: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with (
        input_path.open(
            "r",
            encoding="utf-8",
        ) as input_file,
        output_path.open(
            "w",
            encoding="utf-8",
        ) as output_file,
    ):
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

                converted = build_transition(
                    record
                )

            except Exception as exc:
                raise ValueError(
                    f"failed to process line "
                    f"{line_number}: {exc}"
                ) from exc

            output_file.write(
                json.dumps(
                    converted,
                    ensure_ascii=False,
                )
                + "\n"
            )

            count += 1

    return count
