from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_feature_data(path: str) -> tuple[list[str], np.ndarray]:
    """JSONLから特徴量差分を読み込む。"""
    rows = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            values = dict(record["feature_deltas"])

            # F40はfeature_deltasの外に保存されている
            if "f40" in record:
                values["F40"] = record["f40"]

            rows.append(values)

    if not rows:
        raise ValueError("特徴量データが存在しません。")

    feature_names = sorted(rows[0].keys())

    matrix = np.array(
        [
            [row[name] for name in feature_names]
            for row in rows
        ],
        dtype=np.float64,
    )

    return feature_names, matrix


def compare_features(
    feature_names: list[str],
    matrix: np.ndarray,
) -> list[dict]:
    """特徴量同士の完全一致・最大差・平均差を調べる。"""

    results = []

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            name_a = feature_names[i]
            name_b = feature_names[j]

            a = matrix[:, i]
            b = matrix[:, j]

            diff = np.abs(a - b)

            max_abs_diff = float(np.max(diff))
            mean_abs_diff = float(np.mean(diff))

            exact_match_count = int(np.sum(diff == 0.0))
            exact_match_rate = exact_match_count / len(diff)

            correlation = float(np.corrcoef(a, b)[0, 1])

            results.append(
                {
                    "feature_a": name_a,
                    "feature_b": name_b,
                    "correlation": correlation,
                    "max_abs_diff": max_abs_diff,
                    "mean_abs_diff": mean_abs_diff,
                    "exact_match_count": exact_match_count,
                    "exact_match_rate": exact_match_rate,
                }
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="特徴量間の完全重複・冗長性を確認します。"
    )

    parser.add_argument(
        "--input",
        default="data/processed/feature_transitions_wcsc36.jsonl",
    )

    parser.add_argument(
        "--output",
        default="data/processed/feature_redundancy_wcsc36.json",
    )

    args = parser.parse_args()

    feature_names, matrix = load_feature_data(args.input)

    results = compare_features(feature_names, matrix)

    output = {
        "input": args.input,
        "position_count": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
        "pairs": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("特徴量冗長性チェック完了")
    print("------------------------------")
    print(f"局面数   : {matrix.shape[0]}")
    print(f"特徴量数 : {matrix.shape[1]}")
    print(f"出力     : {output_path}")
    print()

    # 完全一致
    print("【完全一致する特徴量】")
    print("------------------------------")

    exact_pairs = [
        result
        for result in results
        if result["exact_match_rate"] == 1.0
    ]

    if not exact_pairs:
        print("なし")
    else:
        for result in exact_pairs:
            print(
                f'{result["feature_a"]} - {result["feature_b"]} '
                f'一致率={result["exact_match_rate"]:.1%} '
                f'最大差={result["max_abs_diff"]:.10f}'
            )

    print()

    # ほぼ完全一致
    print("【ほぼ完全一致する特徴量】")
    print("------------------------------")

    near_pairs = [
        result
        for result in results
        if result["exact_match_rate"] >= 0.99
        and result["exact_match_rate"] < 1.0
    ]

    if not near_pairs:
        print("なし")
    else:
        near_pairs.sort(
            key=lambda x: x["exact_match_rate"],
            reverse=True,
        )

        for result in near_pairs:
            print(
                f'{result["feature_a"]} - {result["feature_b"]} '
                f'一致率={result["exact_match_rate"]:.2%} '
                f'最大差={result["max_abs_diff"]:.10f} '
                f'平均差={result["mean_abs_diff"]:.10f}'
            )

    print()

    # 高相関
    print("【高相関特徴量】")
    print("------------------------------")

    high_pairs = [
        result
        for result in results
        if abs(result["correlation"]) >= 0.8
    ]

    high_pairs.sort(
        key=lambda x: abs(x["correlation"]),
        reverse=True,
    )

    if not high_pairs:
        print("なし")
    else:
        for result in high_pairs:
            print(
                f'{result["feature_a"]} - {result["feature_b"]} '
                f'r={result["correlation"]:+.4f} '
                f'一致率={result["exact_match_rate"]:.2%}'
            )


if __name__ == "__main__":
    main()