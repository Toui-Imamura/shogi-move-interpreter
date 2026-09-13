from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_feature_data(path: str | Path):
    path = Path(path)

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            values = dict(record["feature_deltas"])

            if "f40" in record:
                values["F40"] = record["f40"]

            rows.append(values)

    return rows


def main():
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
            "feature_correlation_wcsc36.json"
        ),
    )

    args = parser.parse_args()

    rows = load_feature_data(args.input)

    feature_names = sorted(rows[0].keys())

    matrix = np.array(
        [
            [row[name] for name in feature_names]
            for row in rows
        ],
        dtype=np.float64,
    )

    correlation = np.corrcoef(
        matrix,
        rowvar=False,
    )

    result = {
        "feature_names": feature_names,
        "position_count": len(rows),
        "pearson_correlation": {
            feature_names[i]: {
                feature_names[j]: float(
                    correlation[i, j]
                )
                for j in range(len(feature_names))
            }
            for i in range(len(feature_names))
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("相関分析完了")
    print("------------------------------")
    print(f"局面数: {len(rows)}")
    print(f"特徴量数: {len(feature_names)}")
    print(f"出力: {output_path}")

    print()
    print("高相関ペア:")
    print("------------------------------")

    pairs = []

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            r = correlation[i, j]

            if abs(r) >= 0.8:
                pairs.append(
                    (
                        feature_names[i],
                        feature_names[j],
                        r,
                    )
                )

    pairs.sort(
        key=lambda x: abs(x[2]),
        reverse=True,
    )

    for a, b, r in pairs:
        print(
            f"{a:>3} - {b:<3}: "
            f"{r:+.4f}"
        )


if __name__ == "__main__":
    main()