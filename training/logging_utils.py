"""
training/logging_utils.py

機械学習の実験ログを保存するためのユーティリティ。

保存対象:
- 実験設定 config.json
- 学習指標 metrics.jsonl
- 特徴量重み weights.jsonl
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


FEATURE_NAMES = tuple(f"F{i:02d}" for i in range(1, 41))


def _json_default(value: Any) -> Any:
    """
    JSONで扱えない代表的な型を変換する。
    """
    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        return value.item()

    if hasattr(value, "tolist"):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """
    JSON Lines形式で1レコードを追記する。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        json.dump(
            dict(record),
            file,
            ensure_ascii=False,
            default=_json_default,
            allow_nan=False,
        )
        file.write("\n")


def create_run_directory(
    root_dir: str | Path = "data/training_logs",
    run_name: str | None = None,
) -> Path:
    """
    学習実験用のディレクトリを作成する。

    run_nameを指定しない場合:
        run_YYYYMMDD_HHMMSS

    同名ディレクトリが存在する場合は、
    _001, _002 のような連番を付ける。
    """
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    run_dir = root / run_name

    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        return run_dir

    index = 1

    while True:
        candidate = root / f"{run_name}_{index:03d}"

        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate

        index += 1


def save_config(
    run_dir: str | Path,
    config: Mapping[str, Any],
) -> Path:
    """
    実験設定をconfig.jsonとして保存する。
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    output_path = run_path / "config.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            dict(config),
            file,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
            allow_nan=False,
        )
        file.write("\n")

    return output_path


def log_metrics(
    run_dir: str | Path,
    epoch: int,
    metrics: Mapping[str, float | int],
) -> Path:
    """
    学習指標をmetrics.jsonlに追記する。

    保存例:
    {
      "timestamp": "...",
      "epoch": 1,
      "train_loss": 0.8,
      "validation_loss": 0.9,
      "learning_rate": 0.001
    }
    """
    run_path = Path(run_dir)

    record: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "epoch": int(epoch),
    }

    for name, value in metrics.items():
        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise ValueError(
                f"Metric '{name}' must be finite, got {value}"
            )

        record[name] = numeric_value

    output_path = run_path / "metrics.jsonl"
    _append_jsonl(output_path, record)

    return output_path


def normalize_feature_weights(
    weights: Mapping[str, float] | Sequence[float],
) -> dict[str, float]:
    """
    特徴量重みをF01～F40の辞書に変換する。

    入力例1:
        [0.1, 0.2, ...]

    入力例2:
        {"F01": 0.1, "F02": 0.2, ...}

    重みの長さが40でない場合はエラーにする。
    """
    if isinstance(weights, Mapping):
        unknown_names = set(weights) - set(FEATURE_NAMES)

        if unknown_names:
            raise ValueError(
                f"Unknown feature names: {sorted(unknown_names)}"
            )

        result: dict[str, float] = {}

        for feature_name in FEATURE_NAMES:
            value = float(weights.get(feature_name, 0.0))

            if not math.isfinite(value):
                raise ValueError(
                    f"Weight for '{feature_name}' must be finite, got {value}"
                )

            result[feature_name] = value

        return result

    values = list(weights)

    if len(values) != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} weights, got {len(values)}"
        )

    result = {}

    for feature_name, value in zip(FEATURE_NAMES, values):
        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            raise ValueError(
                f"Weight for '{feature_name}' must be finite, got {value}"
            )

        result[feature_name] = numeric_value

    return result


def log_weights(
    run_dir: str | Path,
    epoch: int,
    weights: Mapping[str, float] | Sequence[float],
) -> Path:
    """
    F01～F40の重みをweights.jsonlに追記する。

    保存例:
    {
      "timestamp": "...",
      "epoch": 1,
      "weights": {
        "F01": 0.1,
        "F02": 0.2
      }
    }
    """
    run_path = Path(run_dir)
    normalized_weights = normalize_feature_weights(weights)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "epoch": int(epoch),
        "weights": normalized_weights,
    }

    output_path = run_path / "weights.jsonl"
    _append_jsonl(output_path, record)

    return output_path


def load_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    """
    JSON Linesファイルを読み込む。
    """
    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {input_path}:{line_number}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object at {input_path}:{line_number}"
                )

            records.append(record)

    return records
