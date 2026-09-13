from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev


def percentile(
    values: list[float],
    p: float,
) -> float:
    """
    線形補間によるパーセンタイル。
    pは0〜100。
    """

    if not values:
        return float("nan")

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * (p / 100.0)

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    weight = position - lower

    return (
        values[lower] * (1.0 - weight)
        + values[upper] * weight
    )


def analyze(
    input_path: str | Path,
    output_path: str | Path,
) -> None:

    input_path = Path(input_path)
    output_path = Path(output_path)

    feature_values: dict[str, list[float]] = {}

    count = 0

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            deltas = record["feature_deltas"]

            for name, value in deltas.items():
                feature_values.setdefault(
                    name,
                    [],
                ).append(float(value))

            if "f40" in record:
                feature_values.setdefault(
                    "F40",
                    [],
                ).append(
                    float(record["f40"])
                )

            count += 1

    statistics = {}

    for name, values in sorted(
        feature_values.items()
    ):

        nonzero = sum(
            1
            for value in values
            if abs(value) > 1e-12
        )

        statistics[name] = {
            "count": len(values),
            "mean": mean(values),
            "std": pstdev(values),
            "min": min(values),
            "p01": percentile(values, 1),
            "p05": percentile(values, 5),
            "p25": percentile(values, 25),
            "median": median(values),
            "p75": percentile(values, 75),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "max": max(values),
            "nonzero_count": nonzero,
            "nonzero_ratio": (
                nonzero / len(values)
                if values
                else 0.0
            ),
        }

    output = {
        "dataset": str(input_path),
        "position_count": count,
        "features": statistics,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("特徴量統計量計算完了")
    print("------------------------------")
    print(f"局面数: {count}")
    print(f"特徴量数: {len(statistics)}")
    print(f"出力: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "data/processed/"
            "feature_transitions_wcsc36.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/processed/"
            "feature_statistics_wcsc36.json"
        ),
    )

    args = parser.parse_args()

    analyze(
        args.input,
        args.output,
    )


if __name__ == "__main__":
    main()