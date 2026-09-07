from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_feature_deltas(
    input_path: str | Path,
) -> dict[str, list[float]]:
    """
    transition_positions.jsonlから
    各特徴量のΔ値を読み込む。
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"input file not found: {input_path}"
        )

    values: dict[str, list[float]] = {}

    with input_path.open(
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
                record: dict[str, Any] = json.loads(line)
                feature_deltas = record[
                    "feature_deltas"
                ]

                for feature_name, value in (
                    feature_deltas.items()
                ):
                    value = float(value)

                    if not math.isfinite(value):
                        continue

                    values.setdefault(
                        feature_name,
                        [],
                    ).append(value)

            except Exception as exc:
                raise ValueError(
                    f"failed to process line "
                    f"{line_number}: {exc}"
                ) from exc

    return values


def percentile(
    values: list[float],
    percentage: float,
) -> float:
    """
    線形補間によるパーセンタイル。
    """

    if not values:
        raise ValueError(
            "values must not be empty"
        )

    if not 0 <= percentage <= 100:
        raise ValueError(
            "percentage must be between 0 and 100"
        )

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (
        percentage / 100
    ) * (len(sorted_values) - 1)

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return sorted_values[lower]

    weight = position - lower

    return (
        sorted_values[lower]
        * (1.0 - weight)
        + sorted_values[upper]
        * weight
    )


def summarize_feature(
    values: list[float],
) -> dict[str, float | int]:
    """
    1特徴量の統計量を計算する。
    """

    if not values:
        raise ValueError(
            "values must not be empty"
        )

    mean = statistics.fmean(values)

    if len(values) >= 2:
        std = statistics.stdev(values)
    else:
        std = 0.0

    return {
        "count": len(values),
        "mean": mean,
        "std": std,
        "min": min(values),
        "p01": percentile(values, 1),
        "p05": percentile(values, 5),
        "p25": percentile(values, 25),
        "median": percentile(values, 50),
        "p75": percentile(values, 75),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
        "mean_abs": statistics.fmean(
            abs(value)
            for value in values
        ),
    }


def analyze_feature_distribution(
    input_path: str | Path,
) -> dict[str, dict[str, float | int]]:
    """
    全特徴量の統計量を計算する。
    """

    feature_values = load_feature_deltas(
        input_path
    )

    result: dict[
        str,
        dict[str, float | int],
    ] = {}

    for feature_name in sorted(
        feature_values
    ):
        result[feature_name] = (
            summarize_feature(
                feature_values[feature_name]
            )
        )

    return result


def save_summary_json(
    summary: dict[str, dict[str, float | int]],
    output_path: str | Path,
) -> None:
    """
    統計量をJSONとして保存する。
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )


def print_summary(
    summary: dict[str, dict[str, float | int]],
) -> None:
    """
    統計量をコンソールへ表示する。
    """

    print(
        "feature  "
        "count       mean        std"
        "        min        p05"
        "    median        p95        max"
    )

    print("-" * 100)

    for feature_name, stats in summary.items():
        print(
            f"{feature_name:>5} "
            f"{stats['count']:>7} "
            f"{stats['mean']:>11.6f} "
            f"{stats['std']:>11.6f} "
            f"{stats['min']:>11.6f} "
            f"{stats['p05']:>11.6f} "
            f"{stats['median']:>11.6f} "
            f"{stats['p95']:>11.6f} "
            f"{stats['max']:>11.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze feature-delta distributions."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Path to transition_positions.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to save summary JSON"
        ),
    )

    args = parser.parse_args()

    summary = analyze_feature_distribution(
        args.input
    )

    print_summary(summary)

    if args.output is not None:
        save_summary_json(
            summary,
            args.output,
        )

        print(
            f"\n統計量を保存しました: "
            f"{args.output}"
        )


if __name__ == "__main__":
    main()
