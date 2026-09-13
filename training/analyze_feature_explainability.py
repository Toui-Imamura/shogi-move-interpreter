"""
analyze_feature_explainability.py

特徴量が指し手の説明に利用できるかを調査するための分析。

確認内容:
- 特徴量ごとの変化回数
- 増加・減少・変化なしの割合
- 平均・標準偏差・最小・最大
- 非ゼロ率
- 変化方向の偏り
- 同時に変化する特徴量
- 代表的な遷移レコード

対応形式:
{
    "feature_deltas": {
        "F01": 0.0,
        ...
        "F33": 0.0
    },
    "f40": 8.0
}

注意:
- NNUE評価値との相関だけで特徴量の採否を決定しない
- この分析は説明可能性、実装妥当性、変化の明確さを確認するために使用する
- 出力されていない特徴量は「未実装」と断定せず、別途確認する
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EPSILON = 1e-12


def safe_float(value: Any) -> float | None:
    """数値へ変換できる場合はfloatを返す。"""
    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result):
        return None

    return result


def feature_sort_key(name: str) -> tuple[int, str]:
    """
    F01, F02, ..., F40 の順に並べる。
    """
    normalized = name.upper()

    if normalized.startswith("F") and normalized[1:].isdigit():
        return int(normalized[1:]), normalized

    return 9999, normalized


def mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def stddev(values: list[float], avg: float | None = None) -> float:
    if not values:
        return 0.0

    if avg is None:
        avg = mean(values)

    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def percentile(sorted_values: list[float], p: float) -> float:
    """線形補間によるパーセンタイル。"""
    if not sorted_values:
        return 0.0

    if len(sorted_values) == 1:
        return sorted_values[0]

    index = (len(sorted_values) - 1) * p
    lower = int(math.floor(index))
    upper = int(math.ceil(index))

    if lower == upper:
        return sorted_values[lower]

    weight = index - lower

    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def normalize_feature_name(name: str) -> str | None:
    """
    特徴量名をF01～F40形式へ正規化する。

    例:
    F01 -> F01
    f01 -> F01
    f40 -> F40
    """
    if not isinstance(name, str):
        return None

    name = name.strip().upper()

    if not name.startswith("F"):
        return None

    suffix = name[1:]

    if not suffix.isdigit():
        return None

    number = int(suffix)

    if number < 1 or number > 40:
        return None

    return f"F{number:02d}"


def extract_feature_deltas(record: dict[str, Any]) -> dict[str, float]:
    """
    レコードから特徴量の変化量を抽出する。

    対応形式:
    1. feature_deltas 内にF01～F40がある形式
    2. トップレベルにF01～F40がある形式
    3. トップレベルにf40がある形式
    """
    result: dict[str, float] = {}

    # まず feature_deltas 内を確認する
    feature_deltas = record.get("feature_deltas")

    if isinstance(feature_deltas, dict):
        for raw_name, raw_value in feature_deltas.items():
            feature_name = normalize_feature_name(raw_name)

            if feature_name is None:
                continue

            numeric_value = safe_float(raw_value)

            if numeric_value is not None:
                result[feature_name] = numeric_value

    # トップレベルのF01～F40も確認する
    for raw_name, raw_value in record.items():
        feature_name = normalize_feature_name(raw_name)

        if feature_name is None:
            continue

        numeric_value = safe_float(raw_value)

        if numeric_value is not None:
            result[feature_name] = numeric_value

    return result


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"警告: JSON解析失敗 line={line_number}: {exc}"
                )
                continue

            if isinstance(record, dict):
                records.append(record)

    return records


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    feature_values: dict[str, list[float]] = defaultdict(list)
    feature_abs_values: dict[str, list[float]] = defaultdict(list)
    feature_signs: dict[str, Counter[str]] = defaultdict(Counter)
    feature_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    feature_names: set[str] = set()

    for record in records:
        deltas = extract_feature_deltas(record)

        for feature_name, value in deltas.items():
            feature_names.add(feature_name)

            feature_values[feature_name].append(value)
            feature_abs_values[feature_name].append(abs(value))

            if value > EPSILON:
                feature_signs[feature_name]["positive"] += 1
            elif value < -EPSILON:
                feature_signs[feature_name]["negative"] += 1
            else:
                feature_signs[feature_name]["zero"] += 1

            feature_records[feature_name].append(
                {
                    "game_id": record.get("game_id"),
                    "ply": record.get("ply"),
                    "move": record.get("move"),
                    "sfen": record.get("sfen"),
                    "value": value,
                }
            )

    feature_statistics: dict[str, Any] = {}

    for feature_name in sorted(feature_names, key=feature_sort_key):
        values = feature_values[feature_name]
        abs_values = feature_abs_values[feature_name]
        signs = feature_signs[feature_name]

        sorted_values = sorted(values)
        nonzero_values = [
            value for value in values
            if abs(value) > EPSILON
        ]

        avg = mean(values)
        abs_avg = mean(abs_values)

        examples = sorted(
            feature_records[feature_name],
            key=lambda item: abs(float(item["value"])),
            reverse=True,
        )[:5]

        feature_statistics[feature_name] = {
            "record_count": len(values),
            "nonzero_count": len(nonzero_values),
            "nonzero_rate": (
                len(nonzero_values) / len(values)
                if values
                else 0.0
            ),
            "positive_count": signs["positive"],
            "negative_count": signs["negative"],
            "zero_count": signs["zero"],
            "positive_rate": (
                signs["positive"] / len(values)
                if values
                else 0.0
            ),
            "negative_rate": (
                signs["negative"] / len(values)
                if values
                else 0.0
            ),
            "mean": avg,
            "absolute_mean": abs_avg,
            "stddev": stddev(values, avg),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
            "p05": percentile(sorted_values, 0.05),
            "p25": percentile(sorted_values, 0.25),
            "median": percentile(sorted_values, 0.50),
            "p75": percentile(sorted_values, 0.75),
            "p95": percentile(sorted_values, 0.95),
            "representative_examples": examples,
        }

    # 同時に変化する特徴量の組み合わせを数える
    pair_counts: Counter[tuple[str, str]] = Counter()

    for record in records:
        deltas = extract_feature_deltas(record)

        active_features = sorted(
            [
                feature_name
                for feature_name, value in deltas.items()
                if abs(value) > EPSILON
            ],
            key=feature_sort_key,
        )

        for index, feature_a in enumerate(active_features):
            for feature_b in active_features[index + 1:]:
                pair_counts[(feature_a, feature_b)] += 1

    pair_statistics = [
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "cooccurrence_count": count,
            "cooccurrence_rate": (
                count / len(records)
                if records
                else 0.0
            ),
        }
        for (feature_a, feature_b), count in pair_counts.most_common()
    ]

    # 仕様上のF01～F40のうち、今回のデータに存在しないもの
    expected_features = {
        f"F{number:02d}"
        for number in range(1, 41)
    }

    missing_features = sorted(
        expected_features - feature_names,
        key=feature_sort_key,
    )

    return {
        "record_count": len(records),
        "feature_count": len(feature_statistics),
        "observed_features": sorted(
            feature_statistics.keys(),
            key=feature_sort_key,
        ),
        "missing_features": missing_features,
        "features": feature_statistics,
        "cooccurring_pairs": pair_statistics,
    }


def print_summary(result: dict[str, Any]) -> None:
    print("特徴量説明適性分析完了")
    print("------------------------------")
    print(f"レコード数 : {result['record_count']}")
    print(f"特徴量数   : {result['feature_count']}")

    print()
    print("出力された特徴量")
    print("------------------------------")
    print(", ".join(result["observed_features"]))

    print()
    print("今回のデータに存在しない特徴量")
    print("------------------------------")
    print(", ".join(result["missing_features"]))

    print()
    print("特徴量統計")
    print("------------------------------")
    print(
        "特徴量     非ゼロ率   平均絶対変化量   "
        "正変化率   負変化率   標準偏差"
    )

    features = result["features"]

    for feature_name in sorted(features, key=feature_sort_key):
        item = features[feature_name]

        print(
            f"{feature_name:6s} "
            f"{item['nonzero_rate']:9.4f} "
            f"{item['absolute_mean']:15.6f} "
            f"{item['positive_rate']:9.4f} "
            f"{item['negative_rate']:9.4f} "
            f"{item['stddev']:10.6f}"
        )

    print()
    print("同時変化が多い特徴量ペア")
    print("------------------------------")

    for item in result["cooccurring_pairs"][:20]:
        print(
            f"{item['feature_a']} - {item['feature_b']}: "
            f"{item['cooccurrence_count']}件 "
            f"({item['cooccurrence_rate']:.4f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="特徴量の説明適性・変化分布を分析する"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="特徴量遷移データJSONL",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="分析結果JSON",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    records = load_records(input_path)
    result = analyze(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print_summary(result)
    print()
    print(f"出力 : {output_path}")


if __name__ == "__main__":
    main()