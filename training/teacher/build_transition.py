from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_transitions(
    input_path: str | Path,
    output_path: str | Path,
) -> None:

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"入力ファイルが見つかりません: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    transition_count = 0
    skipped_boundary = 0

    with (
        input_path.open("r", encoding="utf-8") as infile,
        output_path.open("w", encoding="utf-8") as outfile,
    ):
        previous = None

        for line_number, line in enumerate(
            infile,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            current = json.loads(line)

            if previous is not None:

                # 同一ゲーム内だけで遷移を作る。
                if (
                    previous["game_id"]
                    == current["game_id"]
                ):

                    current_score = previous[
                        "teacher_score_cp_black"
                    ]

                    next_score = current[
                        "teacher_score_cp_black"
                    ]

                    delta_black = (
                        next_score - current_score
                    )

                    side_to_move = (
                        previous["sfen"].split()[1]
                    )

                    if side_to_move == "b":
                        delta_mover = delta_black

                    elif side_to_move == "w":
                        delta_mover = -delta_black

                    else:
                        raise ValueError(
                            f"不正な手番: {side_to_move}"
                        )

                    transition = {
                        "game_id": previous["game_id"],
                        "ply": previous["ply"],
                        "sfen": previous["sfen"],
                        "move": previous["move"],
                        "result": previous["result"],
                        "source": previous.get("source"),

                        "side_to_move": side_to_move,

                        "teacher_score_before_black_cp":
                            current_score,

                        "teacher_score_after_black_cp":
                            next_score,

                        "teacher_delta_black_cp":
                            delta_black,

                        "teacher_delta_mover_cp":
                            delta_mover,

                        "teacher_bestmove":
                            previous["teacher_bestmove"],

                        "teacher_depth":
                            previous["teacher_depth"],

                        "teacher_seldepth":
                            previous.get("teacher_seldepth"),
                    }

                    outfile.write(
                        json.dumps(
                            transition,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    transition_count += 1

                else:
                    skipped_boundary += 1

            previous = current

    print()
    print("教師遷移データ生成完了")
    print("------------------------------")
    print(f"入力     : {input_path}")
    print(f"出力     : {output_path}")
    print(f"遷移数   : {transition_count}")
    print(f"境界除外 : {skipped_boundary}")


def main():
    parser = argparse.ArgumentParser(
        description="教師評価値から指し手前後の評価値変化を生成"
    )

    parser.add_argument(
        "--input",
        default="data/processed/teacher_values.jsonl",
    )

    parser.add_argument(
        "--output",
        default="data/processed/teacher_transitions.jsonl",
    )

    args = parser.parse_args()

    build_transitions(
        args.input,
        args.output,
    )


if __name__ == "__main__":
    main()
