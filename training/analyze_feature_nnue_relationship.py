from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON解析エラー: {path}:{line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"レコードがオブジェクトではありません: "
                    f"{path}:{line_number}"
                )

            records.append(record)

    return records


def make_key(record: dict[str, Any]) -> tuple[Any, Any]:
    return (
        record.get("game_id"),
        record.get("ply"),
    )


def pearson_correlation(
    x_values: list[float],
    y_values: list[float],
) -> float | None:
    if len(x_values) < 2 or len(y_values) < 2:
        return None

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

    if len(x) < 2:
        return None

    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None

    return float(np.corrcoef(x, y)[0, 1])


def sign_agreement_rate(
    x_values: list[float],
    y_values: list[float],
    epsilon: float = 1e-12,
) -> float | None:
    if not x_values:
        return None

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return None

    x_sign = np.where(
        x > epsilon,
        1,
        np.where(x < -epsilon, -1, 0),
    )
    y_sign = np.where(
        y > epsilon,
        1,
        np.where(y < -epsilon, -1, 0),
    )

    return float(np.mean(x_sign == y_sign))


def direction_agreement_rate_nonzero(
    x_values: list[float],
    y_values: list[float],
    epsilon: float = 1e-12,
) -> float | None:
    """
    両方の変化量が0ではないデータだけを対象に、
    増減方向が一致する割合を計算する。
    """
    if not x_values:
        return None

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & (np.abs(x) > epsilon)
        & (np.abs(y) > epsilon)
    )

    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return None

    return float(np.mean(np.sign(x) == np.sign(y)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "特徴量変化とNNUE評価値変化の関係を分析する"
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="特徴量遷移データJSONL",
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        required=True,
        help="NNUE教師遷移データJSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="分析結果JSON",
    )
    parser.add_argument(
        "--target",
        choices=[
            "teacher_delta_black_cp",
            "teacher_delta_mover_cp",
        ],
        default="teacher_delta_black_cp",
        help="相関分析対象のNNUE評価値変化",
    )

    args = parser.parse_args()

    feature_records = load_jsonl(args.features)
    teacher_records = load_jsonl(args.teacher)

    teacher_by_key: dict[tuple[Any, Any], dict[str, Any]] = {}

    for record in teacher_records:
        key = make_key(record)
        teacher_by_key[key] = record

    feature_names: set[str] = set()
    joined_records: list[dict[str, Any]] = []

    for feature_record in feature_records:
        key = make_key(feature_record)
        teacher_record = teacher_by_key.get(key)

        if teacher_record is None:
            continue

        feature_deltas = feature_record.get("feature_deltas", {})

        if not isinstance(feature_deltas, dict):
            continue

        for feature_name in feature_deltas:
            feature_names.add(feature_name)

        joined_records.append(
            {
                "game_id": feature_record.get("game_id"),
                "ply": feature_record.get("ply"),
                "move": feature_record.get("move"),
                "feature_deltas": feature_deltas,
                "f40": feature_record.get("f40"),
                "teacher_target": teacher_record.get(args.target),
            }
        )

    feature_names_sorted = sorted(feature_names)

    analysis: dict[str, Any] = {
        "feature_input": str(args.features),
        "teacher_input": str(args.teacher),
        "target": args.target,
        "feature_record_count": len(feature_records),
        "teacher_record_count": len(teacher_records),
        "joined_record_count": len(joined_records),
        "feature_count": len(feature_names_sorted),
        "features": {},
    }

    for feature_name in feature_names_sorted:
        x_values: list[float] = []
        y_values: list[float] = []

        for record in joined_records:
            feature_deltas = record["feature_deltas"]
            x_value = feature_deltas.get(feature_name)
            y_value = record.get("teacher_target")

            if x_value is None or y_value is None:
                continue

            try:
                x = float(x_value)
                y = float(y_value)
            except (TypeError, ValueError):
                continue

            if not np.isfinite(x) or not np.isfinite(y):
                continue

            x_values.append(x)
            y_values.append(y)

        correlation = pearson_correlation(x_values, y_values)
        agreement = sign_agreement_rate(x_values, y_values)
        agreement_nonzero = direction_agreement_rate_nonzero(
            x_values,
            y_values,
        )

        analysis["features"][feature_name] = {
            "sample_count": len(x_values),
            "pearson_r": correlation,
            "sign_agreement_rate": agreement,
            "nonzero_direction_agreement_rate": agreement_nonzero,
            "feature_mean": (
                float(np.mean(x_values))
                if x_values
                else None
            ),
            "feature_std": (
                float(np.std(x_values))
                if x_values
                else None
            ),
            "teacher_target_mean": (
                float(np.mean(y_values))
                if y_values
                else None
            ),
            "teacher_target_std": (
                float(np.std(y_values))
                if y_values
                else None
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(
            analysis,
            f,
            ensure_ascii=False,
            indent=2,
        )

    ranked = [
        (
            name,
            values["pearson_r"],
            values["sample_count"],
        )
        for name, values in analysis["features"].items()
        if values["pearson_r"] is not None
    ]

    ranked.sort(
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    print("特徴量-NNUE関係分析完了")
    print("-" * 30)
    print(f"特徴量レコード数 : {len(feature_records)}")
    print(f"教師レコード数   : {len(teacher_records)}")
    print(f"結合レコード数   : {len(joined_records)}")
    print(f"特徴量数         : {len(feature_names_sorted)}")
    print(f"対象             : {args.target}")
    print(f"出力             : {args.output}")

    print()
    print("相関係数の絶対値が大きい特徴量")
    print("-" * 30)

    for name, correlation, sample_count in ranked[:10]:
        print(
            f"{name}: "
            f"r={correlation:+.4f}, "
            f"n={sample_count}"
        )


if __name__ == "__main__":
    main()
