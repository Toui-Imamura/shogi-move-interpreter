from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cshogi

from interpreter.feature_adapters import (
    compute_f01_f33_deltas,
)
from features.move_transition import (
    compute_move_transition,
)


def build_feature_transition(
    record: dict[str, Any],
) -> dict[str, Any]:
    """
    1局面について、実際の指し手による
    F01〜F33の変化量とF40を計算する。

    cshogi 1.0.4で確認済みのAPIのみ使用する。
    """

    if "sfen" not in record:
        raise ValueError("record must contain 'sfen'")

    if "move" not in record:
        raise ValueError("record must contain 'move'")

    # --------------------------------------------------
    # 指し手前
    # --------------------------------------------------
    before = cshogi.Board(
        sfen=record["sfen"]
    )

    # --------------------------------------------------
    # USI → cshogi内部の指し手
    # cshogi 1.0.4:
    #   board.move_from_usi() は存在する
    # --------------------------------------------------
    move = before.move_from_usi(
        record["move"]
    )

    if not before.is_legal(move):
        raise ValueError(
            f"illegal move: {record['move']}"
        )

    # --------------------------------------------------
    # 指し手後
    # --------------------------------------------------
    after = before.copy()
    after.push(move)

    # --------------------------------------------------
    # F01〜F33
    # --------------------------------------------------
    feature_deltas = compute_f01_f33_deltas(
        before,
        after,
        move=move,
    )

    # --------------------------------------------------
    # F40
    #
    # F37〜F39はMCTS情報が必要なので、
    # 今回の統計処理には入れない。
    # --------------------------------------------------
    f40_result = compute_move_transition(
        before,
        after,
        feature_deltas,
    )

    return {
        "game_id": record.get("game_id"),
        "ply": record.get("ply"),
        "sfen": record["sfen"],
        "move": record["move"],
        "result": record.get("result"),
        "feature_deltas": feature_deltas,
        "f40": float(f40_result.degree),
    }


def convert_dataset(
    input_path: str | Path,
    output_path: str | Path,
) -> int:
    """
    WCSC36 positions JSONLを読み込み、
    F01〜F33の変化量とF40を計算する。
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

                converted = build_feature_transition(
                    record
                )

            except Exception as exc:
                raise RuntimeError(
                    f"failed at line {line_number}: {exc}"
                ) from exc

            output_file.write(
                json.dumps(
                    converted,
                    ensure_ascii=False,
                )
                + "\n"
            )

            count += 1

            if count % 1000 == 0:
                print(
                    f"処理済み: {count}局面",
                    flush=True,
                )

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute F01-F33 deltas and F40 "
            "for WCSC36 positions."
        )
    )

    parser.add_argument(
        "--input",
        default="data/processed/positions_wcsc36.jsonl",
    )

    parser.add_argument(
        "--output",
        default=(
            "data/processed/"
            "feature_transitions_wcsc36.jsonl"
        ),
    )

    args = parser.parse_args()

    count = convert_dataset(
        args.input,
        args.output,
    )

    print()
    print("特徴量変換完了")
    print("------------------------------")
    print(f"入力 : {args.input}")
    print(f"出力 : {args.output}")
    print(f"局面数: {count}")


if __name__ == "__main__":
    main()