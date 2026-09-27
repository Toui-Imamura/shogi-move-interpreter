from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


# scripts/から実行した場合でも、リポジトリルートをimport対象にする
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.feature_weight_dataset import load_training_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WCSC36 295局用の特徴量重み学習データを生成する"
    )

    parser.add_argument(
        "--teacher",
        type=Path,
        default=Path(
            "data/processed/teacher_transitions_wcsc36_295games.jsonl"
        ),
        help="教師遷移データ",
    )

    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "data/processed/feature_transitions_wcsc36.jsonl"
        ),
        help="特徴量遷移データ",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/feature_weight_dataset_wcsc36_295games.jsonl"
        ),
        help="出力先",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.teacher.exists():
        raise FileNotFoundError(
            f"教師遷移データが見つかりません: {args.teacher}"
        )

    if not args.features.exists():
        raise FileNotFoundError(
            f"特徴量遷移データが見つかりません: {args.features}"
        )

    print("学習レコードを読み込んでいます...")

    records = load_training_records(
        teacher_path=args.teacher,
        feature_path=args.features,
        strict=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        for record in records:
            # TrainingRecord dataclassをJSON保存可能な辞書へ変換する
            record_dict = asdict(record)

            file.write(
                json.dumps(
                    record_dict,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("学習データ生成完了")
    print("-" * 30)
    print(f"入力教師データ  : {args.teacher}")
    print(f"入力特徴量データ: {args.features}")
    print(f"出力            : {args.output}")
    print(f"レコード数      : {len(records)}")


if __name__ == "__main__":
    main()
