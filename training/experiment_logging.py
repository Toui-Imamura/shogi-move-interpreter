"""
training/experiment_logging.py

研究用の学習実験ログを保存するためのユーティリティ。

保存する情報:
- 実験メタデータ
- 学習設定
- 特徴量仕様のスナップショット
- JSON / JSONL形式の実験ログ

実験結果は原則として新しいディレクトリへ保存し、
過去の実験結果を上書きしない。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def _json_default(value: Any) -> Any:
    """
    JSONで直接扱えない代表的な型を変換する。

    NumPy型などを保存する場合にも対応しやすいようにする。
    """
    if hasattr(value, "item"):
        return value.item()

    if hasattr(value, "tolist"):
        return value.tolist()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"JSON serializableではない型です: {type(value)!r}"
    )


def write_json(
    path: str | Path,
    data: Mapping[str, Any] | list[Any],
) -> Path:
    """
    JSONファイルをUTF-8で保存する。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        file.write("\n")

    return output_path


def append_jsonl(
    path: str | Path,
    record: Mapping[str, Any],
) -> Path:
    """
    JSON Lines形式で1レコードを追記する。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as file:
        json.dump(
            record,
            file,
            ensure_ascii=False,
            default=_json_default,
        )
        file.write("\n")

    return output_path


def create_experiment_id(
    prefix: str = "exp",
    now: datetime | None = None,
) -> str:
    """
    実験IDを生成する。

    例:
        exp_20260914_143015
    """
    if now is None:
        now = datetime.now().astimezone()

    timestamp = now.strftime("%Y%m%d_%H%M%S")

    safe_prefix = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        prefix,
    ).strip("_")

    if not safe_prefix:
        safe_prefix = "exp"

    return f"{safe_prefix}_{timestamp}"


def create_experiment_directory(
    root_dir: str | Path,
    experiment_id: str,
) -> Path:
    """
    実験ディレクトリを作成する。

    同じ実験IDが存在する場合は上書きせず、エラーにする。
    """
    root_path = Path(root_dir)
    experiment_dir = root_path / experiment_id

    if experiment_dir.exists():
        raise FileExistsError(
            f"実験ディレクトリが既に存在します: {experiment_dir}"
        )

    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def create_unique_experiment_directory(
    root_dir: str | Path,
    experiment_id: str,
) -> tuple[str, Path]:
    """
    実験IDが重複した場合に連番を付けてディレクトリを作成する。

    例:
        exp_20260914_143015
        exp_20260914_143015_001
        exp_20260914_143015_002
    """
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)

    candidate_id = experiment_id
    candidate_dir = root_path / candidate_id
    suffix = 1

    while candidate_dir.exists():
        candidate_id = f"{experiment_id}_{suffix:03d}"
        candidate_dir = root_path / candidate_id
        suffix += 1

    candidate_dir.mkdir(parents=True, exist_ok=False)

    return candidate_id, candidate_dir


def save_experiment_metadata(
    experiment_dir: str | Path,
    metadata: Mapping[str, Any],
) -> Path:
    """
    experiment_metadata.jsonを保存する。
    """
    return write_json(
        Path(experiment_dir) / "experiment_metadata.json",
        dict(metadata),
    )


def save_feature_spec_snapshot(
    experiment_dir: str | Path,
    feature_spec: Mapping[str, Any],
) -> Path:
    """
    feature_spec_snapshot.jsonを保存する。
    """
    return write_json(
        Path(experiment_dir) / "feature_spec_snapshot.json",
        dict(feature_spec),
    )


def save_experiment_config(
    experiment_dir: str | Path,
    config: Mapping[str, Any],
) -> Path:
    """
    config.jsonを保存する。
    """
    return write_json(
        Path(experiment_dir) / "config.json",
        dict(config),
    )


def initialize_experiment(
    root_dir: str | Path,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    feature_spec: Mapping[str, Any],
    experiment_id: str | None = None,
    id_prefix: str = "exp",
) -> tuple[str, Path]:
    """
    実験ディレクトリを作成し、初期ログを一括保存する。

    Returns:
        (実験ID, 実験ディレクトリ)
    """
    if experiment_id is None:
        experiment_id = create_experiment_id(prefix=id_prefix)

    actual_id, experiment_dir = create_unique_experiment_directory(
        root_dir=root_dir,
        experiment_id=experiment_id,
    )

    metadata_with_defaults = dict(metadata)
    metadata_with_defaults.setdefault(
        "experiment_id",
        actual_id,
    )
    metadata_with_defaults.setdefault(
        "created_at",
        datetime.now().astimezone().isoformat(),
    )

    save_experiment_metadata(
        experiment_dir,
        metadata_with_defaults,
    )
    save_experiment_config(
        experiment_dir,
        config,
    )
    save_feature_spec_snapshot(
        experiment_dir,
        feature_spec,
    )

    return actual_id, experiment_dir

def save_metrics_history(
    experiment_dir: str | Path,
    epochs: Iterable[int],
    train_loss: Iterable[float],
    valid_loss: Iterable[float],
    train_mae: Iterable[float],
    valid_mae: Iterable[float],
    train_rmse: Iterable[float],
    valid_rmse: Iterable[float],
) -> Path:
    """
    epochごとの学習指標をmetrics.jsonlへ保存する。

    1行につき1epoch分の記録を保存する。
    """
    output_path = Path(experiment_dir) / "metrics.jsonl"

    rows = zip(
        epochs,
        train_loss,
        valid_loss,
        train_mae,
        valid_mae,
        train_rmse,
        valid_rmse,
    )

    with output_path.open("w", encoding="utf-8") as file:
        for (
            epoch,
            current_train_loss,
            current_valid_loss,
            current_train_mae,
            current_valid_mae,
            current_train_rmse,
            current_valid_rmse,
        ) in rows:
            record = {
                "epoch": int(epoch),
                "train": {
                    "loss": float(current_train_loss),
                    "mae": float(current_train_mae),
                    "rmse": float(current_train_rmse),
                },
                "valid": {
                    "loss": float(current_valid_loss),
                    "mae": float(current_valid_mae),
                    "rmse": float(current_valid_rmse),
                },
            }

            json.dump(
                record,
                file,
                ensure_ascii=False,
                default=_json_default,
                allow_nan=False,
            )
            file.write("\n")

    return output_path


def save_weight_history(
    experiment_dir: str | Path,
    epochs: Iterable[int],
    weight_history: Iterable[Mapping[str, float]],
    bias_history: Iterable[float],
) -> Path:
    """
    epochごとの特徴量重みとバイアスをweight_history.jsonlへ保存する。
    """
    output_path = Path(experiment_dir) / "weight_history.jsonl"

    rows = zip(
        epochs,
        weight_history,
        bias_history,
    )

    with output_path.open("w", encoding="utf-8") as file:
        for epoch, weights, bias in rows:
            record = {
                "epoch": int(epoch),
                "weights": {
                    str(name): float(value)
                    for name, value in weights.items()
                },
                "bias": float(bias),
            }

            json.dump(
                record,
                file,
                ensure_ascii=False,
                default=_json_default,
                allow_nan=False,
            )
            file.write("\n")

    return output_path


def save_validation_predictions(
    experiment_dir: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """
    検証データの予測結果をvalidation_predictions.jsonlへ保存する。
    """
    output_path = Path(experiment_dir) / "validation_predictions.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            json.dump(
                dict(row),
                file,
                ensure_ascii=False,
                default=_json_default,
                allow_nan=False,
            )
            file.write("\n")

    return output_path


def save_metrics_history(
    experiment_dir: str | Path,
    epochs: Iterable[int],
    train_loss: Iterable[float],
    valid_loss: Iterable[float],
    train_mae: Iterable[float],
    valid_mae: Iterable[float],
    train_rmse: Iterable[float],
    valid_rmse: Iterable[float],
) -> Path:
    """
    学習中の評価指標の履歴をmetrics.jsonlへ保存する。

    1行につき1回のログ時点を保存する。
    """
    experiment_path = Path(experiment_dir)
    experiment_path.mkdir(parents=True, exist_ok=True)

    output_path = experiment_path / "metrics.jsonl"

    rows = zip(
        epochs,
        train_loss,
        valid_loss,
        train_mae,
        valid_mae,
        train_rmse,
        valid_rmse,
    )

    with output_path.open("w", encoding="utf-8") as file:
        for (
            epoch,
            train_loss_value,
            valid_loss_value,
            train_mae_value,
            valid_mae_value,
            train_rmse_value,
            valid_rmse_value,
        ) in rows:
            record = {
                "epoch": int(epoch),
                "train_loss": float(train_loss_value),
                "valid_loss": float(valid_loss_value),
                "train_mae": float(train_mae_value),
                "valid_mae": float(valid_mae_value),
                "train_rmse": float(train_rmse_value),
                "valid_rmse": float(valid_rmse_value),
            }
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=_json_default,
                )
                + "\n"
            )

    return output_path


def save_weight_history(
    experiment_dir: str | Path,
    epochs: Iterable[int],
    weight_history: Iterable[Mapping[str, float]],
    bias_history: Iterable[float],
) -> Path:
    """
    学習中の特徴量重みとバイアスの履歴をweight_history.jsonlへ保存する。

    1行につき1回のログ時点を保存する。
    """
    experiment_path = Path(experiment_dir)
    experiment_path.mkdir(parents=True, exist_ok=True)

    output_path = experiment_path / "weight_history.jsonl"

    rows = zip(
        epochs,
        weight_history,
        bias_history,
    )

    with output_path.open("w", encoding="utf-8") as file:
        for epoch, weights, bias in rows:
            record = {
                "epoch": int(epoch),
                "weights": {
                    str(name): float(value)
                    for name, value in weights.items()
                },
                "bias": float(bias),
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=_json_default,
                )
                + "\n"
            )

    return output_path


def save_validation_predictions(
    experiment_dir: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """
    検証データの予測結果をvalidation_predictions.jsonlへ保存する。
    """
    experiment_path = Path(experiment_dir)
    experiment_path.mkdir(parents=True, exist_ok=True)

    output_path = experiment_path / "validation_predictions.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    default=_json_default,
                )
                + "\n"
            )

    return output_path
