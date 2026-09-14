"""
scripts/analyze_feature_weight_dataset.py

特徴量重み学習前のデータ分布を分析する。

入力:
    data/processed/feature_transitions_wcsc36.jsonl
    data/processed/teacher_transitions_wcsc36_10games.jsonl

出力:
    data/processed/feature_weight_dataset_analysis/
        summary.json
        feature_statistics.csv
        target_statistics.json
        feature_target_correlation.csv
        feature_correlation.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

# scripts/から実行した場合でも、
# プロジェクトルートをモジュール検索パスへ追加する。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from training.feature_weight_dataset import (
    FEATURE_NAMES,
    load_training_records,
    records_to_arrays,
)


def _safe_float(value: Any) -> float | None:
    """
    JSONへ保存可能なfloatへ変換する。

    NaNやinfはNoneに変換する。
    """
    value = float(value)

    if not np.isfinite(value):
        return None

    return value


def _statistics(values: np.ndarray) -> dict[str, Any]:
    """
    1次元配列の統計量を計算する。
    """

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "q01": None,
            "q05": None,
            "q25": None,
            "q75": None,
            "q95": None,
            "q99": None,
            "zero_count": 0,
            "nonzero_count": 0,
        }

    return {
        "count": int(values.size),
        "min": _safe_float(np.min(values)),
        "max": _safe_float(np.max(values)),
        "mean": _safe_float(np.mean(values)),
        "median": _safe_float(np.median(values)),
        "std": _safe_float(np.std(values)),
        "q01": _safe_float(np.percentile(values, 1)),
        "q05": _safe_float(np.percentile(values, 5)),
        "q25": _safe_float(np.percentile(values, 25)),
        "q75": _safe_float(np.percentile(values, 75)),
        "q95": _safe_float(np.percentile(values, 95)),
        "q99": _safe_float(np.percentile(values, 99)),
        "zero_count": int(np.count_nonzero(values == 0.0)),
        "nonzero_count": int(np.count_nonzero(values != 0.0)),
    }


def _pearson_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float | None:
    """
    Pearsonの相関係数を計算する。

    分散が0の場合は相関係数を定義できないためNone。
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.size == 0 or y.size == 0:
        return None

    if x.size != y.size:
        raise ValueError(
            "相関計算対象の件数が一致しません。"
        )

    x_std = np.std(x)
    y_std = np.std(y)

    if x_std == 0.0 or y_std == 0.0:
        return None

    correlation = np.corrcoef(x, y)[0, 1]

    return _safe_float(correlation)


def _write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    """
    JSONを書き込む。
    """

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """
    CSVを書き込む。
    """

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def analyze_dataset(
    *,
    teacher_path: str | Path,
    feature_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """
    特徴量重み学習用データを分析する。
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_training_records(
        teacher_path=teacher_path,
        feature_path=feature_path,
        strict=True,
    )

    x, y = records_to_arrays(records)

    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)

    feature_statistics: dict[str, dict[str, Any]] = {}
    feature_statistics_rows: list[dict[str, Any]] = []

    for index, feature_name in enumerate(FEATURE_NAMES):
        stats = _statistics(x64[:, index])
        feature_statistics[feature_name] = stats

        feature_statistics_rows.append(
            {
                "feature": feature_name,
                **stats,
            }
        )

    target_statistics = _statistics(y64)

    feature_target_rows: list[dict[str, Any]] = []

    for index, feature_name in enumerate(FEATURE_NAMES):
        feature_target_rows.append(
            {
                "feature": feature_name,
                "pearson_with_target": (
                    _pearson_correlation(
                        x64[:, index],
                        y64,
                    )
                ),
            }
        )

    correlation_matrix = np.corrcoef(
        x64,
        rowvar=False,
    )

    feature_correlation_rows: list[dict[str, Any]] = []

    for i, feature_name_i in enumerate(FEATURE_NAMES):
        for j, feature_name_j in enumerate(FEATURE_NAMES):
            if j <= i:
                continue

            correlation = correlation_matrix[i, j]

            feature_correlation_rows.append(
                {
                    "feature_i": feature_name_i,
                    "feature_j": feature_name_j,
                    "pearson_correlation": _safe_float(
                        correlation
                    ),
                    "absolute_correlation": _safe_float(
                        abs(correlation)
                    ),
                }
            )

    feature_target_sorted = sorted(
        feature_target_rows,
        key=lambda row: (
            -abs(
                row["pearson_with_target"]
                if row["pearson_with_target"] is not None
                else 0.0
            )
        ),
    )

    feature_correlation_sorted = sorted(
        feature_correlation_rows,
        key=lambda row: -(
            row["absolute_correlation"]
            if row["absolute_correlation"] is not None
            else 0.0
        ),
    )

    summary = {
        "record_count": int(len(records)),
        "feature_count": int(x.shape[1]),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "teacher_path": str(teacher_path),
        "feature_path": str(feature_path),
        "target_statistics": target_statistics,
        "feature_statistics": feature_statistics,
        "feature_target_correlation_sorted": (
            feature_target_sorted
        ),
        "feature_correlation_sorted": (
            feature_correlation_sorted
        ),
    }

    _write_json(
        output_dir / "summary.json",
        summary,
    )

    _write_json(
        output_dir / "target_statistics.json",
        target_statistics,
    )

    _write_csv(
        output_dir / "feature_statistics.csv",
        [
            "feature",
            "count",
            "min",
            "max",
            "mean",
            "median",
            "std",
            "q01",
            "q05",
            "q25",
            "q75",
            "q95",
            "q99",
            "zero_count",
            "nonzero_count",
        ],
        feature_statistics_rows,
    )

    _write_csv(
        output_dir / "feature_target_correlation.csv",
        [
            "feature",
            "pearson_with_target",
        ],
        feature_target_sorted,
    )

    _write_csv(
        output_dir / "feature_correlation.csv",
        [
            "feature_i",
            "feature_j",
            "pearson_correlation",
            "absolute_correlation",
        ],
        feature_correlation_sorted,
    )

    print("=== Dataset analysis ===")
    print(f"record_count: {len(records)}")
    print(f"X shape: {x.shape}")
    print(f"y shape: {y.shape}")

    print("\n=== Target statistics ===")
    for key, value in target_statistics.items():
        print(f"{key}: {value}")

    print("\n=== Feature-target correlation ===")
    for row in feature_target_sorted:
        print(
            f"{row['feature']}: "
            f"{row['pearson_with_target']}"
        )

    print("\n=== Top feature-feature correlations ===")
    for row in feature_correlation_sorted[:20]:
        print(
            f"{row['feature_i']} - "
            f"{row['feature_j']}: "
            f"{row['pearson_correlation']}"
        )

    print("\nOutput directory:")
    print(output_dir)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "特徴量重み学習用データの分布を分析する"
        )
    )

    parser.add_argument(
        "--teacher",
        default=(
            "data/processed/"
            "teacher_transitions_wcsc36_10games.jsonl"
        ),
        help="教師値JSONLのパス",
    )

    parser.add_argument(
        "--features",
        default=(
            "data/processed/"
            "feature_transitions_wcsc36.jsonl"
        ),
        help="特徴量JSONLのパス",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/processed/"
            "feature_weight_dataset_analysis"
        ),
        help="分析結果の出力先",
    )

    args = parser.parse_args()

    analyze_dataset(
        teacher_path=args.teacher,
        feature_path=args.features,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
