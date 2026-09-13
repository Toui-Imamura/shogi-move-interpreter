from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.transition_dataset import build_transition


def load_teacher_records(
    path: str | Path,
) -> dict[tuple[int, int], dict]:
    """
    teacher_transitions.jsonlを読み込み、
    (game_id, ply)をキーとして辞書化する。
    """

    records = {}

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON parse error at line {line_number}"
                ) from exc

            key = (
                int(record["game_id"]),
                int(record["ply"]),
            )

            records[key] = record

    return records


def build_training_dataset(
    positions_path: str | Path,
    teacher_path: str | Path,
    output_path: str | Path,
) -> int:

    positions_path = Path(positions_path)
    teacher_path = Path(teacher_path)
    output_path = Path(output_path)

    if not positions_path.exists():
        raise FileNotFoundError(
            f"positions file not found: {positions_path}"
        )

    if not teacher_path.exists():
        raise FileNotFoundError(
            f"teacher file not found: {teacher_path}"
        )

    teacher_records = load_teacher_records(
        teacher_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0
    skipped = 0

    with (
        positions_path.open(
            "r",
            encoding="utf-8",
        ) as positions_file,

        output_path.open(
            "w",
            encoding="utf-8",
        ) as output_file,
    ):

        for line_number, line in enumerate(
            positions_file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            game_id = int(record["game_id"])
            ply = int(record["ply"])

            key = (game_id, ply)

            teacher = teacher_records.get(key)

            if teacher is None:
                skipped += 1
                continue

            # ------------------------------------------------
            # F01～F33の特徴量変化を計算
            # ------------------------------------------------

            feature_record = build_transition(
                record
            )

            # ------------------------------------------------
            # 教師評価値を結合
            # ------------------------------------------------

            training_record = {
                "game_id": game_id,
                "ply": ply,
                "sfen": record["sfen"],
                "move": record["move"],
                "result": record.get("result"),

                "side_to_move":
                    teacher["side_to_move"],

                # F01～F33
                "feature_deltas":
                    feature_record["feature_deltas"],

                # YaneuraOu教師値
                "teacher_score_before_black_cp":
                    teacher[
                        "teacher_score_before_black_cp"
                    ],

                "teacher_score_after_black_cp":
                    teacher[
                        "teacher_score_after_black_cp"
                    ],

                "teacher_delta_black_cp":
                    teacher[
                        "teacher_delta_black_cp"
                    ],

                "teacher_delta_mover_cp":
                    teacher[
                        "teacher_delta_mover_cp"
                    ],

                # YaneuraOuの推奨手
                "teacher_bestmove":
                    teacher[
                        "teacher_bestmove"
                    ],

                "teacher_depth":
                    teacher[
                        "teacher_depth"
                    ],
            }

            output_file.write(
                json.dumps(
                    training_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            count += 1

    print("学習用データセット生成完了")
    print("------------------------------")
    print(f"入力局面     : {positions_path}")
    print(f"教師データ   : {teacher_path}")
    print(f"出力         : {output_path}")
    print(f"生成件数     : {count}")
    print(f"スキップ件数 : {skipped}")

    return count


def main():
    parser = argparse.ArgumentParser(
        description=(
            "F01～F33とYaneuraOu教師値を結合する"
        )
    )

    parser.add_argument(
        "--positions",
        default="data/processed/positions.jsonl",
    )

    parser.add_argument(
        "--teacher",
        default=(
            "data/processed/"
            "teacher_transitions.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/processed/"
            "training_dataset.jsonl"
        ),
    )

    args = parser.parse_args()

    build_training_dataset(
        positions_path=args.positions,
        teacher_path=args.teacher,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()