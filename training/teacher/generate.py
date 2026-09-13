from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.teacher.yaneuraou import YaneuraOuEvaluator


def count_existing_records(path: Path) -> int:
    if not path.exists():
        return 0

    count = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1

    return count


def generate_teacher_values(
    input_path: str | Path,
    output_path: str | Path,
    engine_path: str | Path,
    eval_dir: str | Path,
    depth: int,
    threads: int,
    hash_mb: int,
    resume: bool = True,
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

    existing_count = (
        count_existing_records(output_path)
        if resume
        else 0
    )

    if existing_count > 0:
        print(
            f"既存教師データ: {existing_count}局面"
        )
        print(
            f"{existing_count + 1}局面目から再開します。"
        )

    evaluator = YaneuraOuEvaluator(
        engine_path=engine_path,
        eval_dir=eval_dir,
        depth=depth,
        threads=threads,
        hash_mb=hash_mb,
    )

    count = 0
    processed = 0

    file_mode = "a" if resume and existing_count > 0 else "w"

    with (
        input_path.open("r", encoding="utf-8") as infile,
        output_path.open(file_mode, encoding="utf-8") as outfile,
    ):
        evaluator.start()

        try:
            for line in infile:
                line = line.strip()

                if not line:
                    continue

                processed += 1

                # 既に生成済みの局面はスキップ。
                if processed <= existing_count:
                    continue

                record = json.loads(line)

                sfen = record["sfen"]

                evaluation = evaluator.evaluate_sfen(sfen)

                teacher_record = {
                    "game_id": record["game_id"],
                    "ply": record["ply"],
                    "sfen": sfen,
                    "move": record["move"],
                    "result": record["result"],
                    "source": record.get("source"),

                    "teacher_score_cp_black": (
                        evaluation["score_cp_black"]
                    ),

                    "teacher_score_cp_side_to_move": (
                        evaluation["score_cp_side_to_move"]
                    ),

                    "teacher_score_raw": (
                        evaluation["score_raw"]
                    ),

                    "teacher_mate": (
                        evaluation["mate"]
                    ),

                    "teacher_depth": (
                        evaluation["depth"]
                    ),

                    "teacher_seldepth": (
                        evaluation["seldepth"]
                    ),

                    "teacher_bestmove": (
                        evaluation["bestmove"]
                    ),
                }

                outfile.write(
                    json.dumps(
                        teacher_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                outfile.flush()

                count += 1

                print(
                    f"[{processed}] "
                    f"ply={record['ply']} "
                    f"score={evaluation['score_raw']} "
                    f"bestmove={evaluation['bestmove']}"
                )

        finally:
            evaluator.close()

    print()
    print("教師データ生成完了")
    print("------------------------------")
    print(f"入力局面 : {input_path}")
    print(f"出力     : {output_path}")
    print(f"今回生成 : {count}")
    print(f"累計     : {existing_count + count}")


def main():
    parser = argparse.ArgumentParser(
        description="YaneuraOuによる教師評価値生成"
    )

    parser.add_argument(
        "--input",
        default="data/processed/positions.jsonl",
    )

    parser.add_argument(
        "--output",
        default="data/processed/teacher_values.jsonl",
    )

    parser.add_argument(
        "--engine",
        default=(
            "/workspace/YaneuraOu/source/"
            "YaneuraOu-by-gcc"
        ),
    )

    parser.add_argument(
        "--eval-dir",
        default="/workspace/YaneuraOu/source/eval",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--hash",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="既存の教師データを無視して最初から生成する",
    )

    args = parser.parse_args()

    generate_teacher_values(
        input_path=args.input,
        output_path=args.output,
        engine_path=args.engine,
        eval_dir=args.eval_dir,
        depth=args.depth,
        threads=args.threads,
        hash_mb=args.hash,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
