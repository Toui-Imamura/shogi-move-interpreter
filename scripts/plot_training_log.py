"""
scripts/plot_training_log.py

学習ログからグラフを生成する。

使用例:
    python scripts/plot_training_log.py data/training_logs/run_xxx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# scripts/ から実行した場合でも、リポジトリルートをimport対象にする。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from training.logging_utils import FEATURE_NAMES, load_jsonl


def _load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"

    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def plot_loss_curve(
    run_dir: Path,
    metrics: list[dict[str, Any]],
) -> Path | None:
    """
    学習損失・検証損失をグラフ化する。
    """
    if not metrics:
        return None

    epochs = [int(record["epoch"]) for record in metrics]

    train_loss = [
        record.get("train_loss")
        for record in metrics
    ]

    validation_loss = [
        record.get("validation_loss")
        for record in metrics
    ]

    has_train_loss = any(value is not None for value in train_loss)
    has_validation_loss = any(
        value is not None for value in validation_loss
    )

    if not has_train_loss and not has_validation_loss:
        return None

    figure, axis = plt.subplots(figsize=(10, 6))

    if has_train_loss:
        axis.plot(
            epochs,
            train_loss,
            marker="o",
            label="Train loss",
        )

    if has_validation_loss:
        axis.plot(
            epochs,
            validation_loss,
            marker="o",
            label="Validation loss",
        )

    axis.set_title("Training loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output_path = run_dir / "loss_curve.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    return output_path


def plot_weight_history(
    run_dir: Path,
    weights_records: list[dict[str, Any]],
) -> Path | None:
    """
    F01～F40の重みの推移をグラフ化する。
    """
    if not weights_records:
        return None

    epochs = [
        int(record["epoch"])
        for record in weights_records
    ]

    figure, axis = plt.subplots(figsize=(14, 8))

    for feature_name in FEATURE_NAMES:
        values = [
            float(record["weights"].get(feature_name, 0.0))
            for record in weights_records
        ]

        axis.plot(
            epochs,
            values,
            label=feature_name,
            linewidth=1.0,
        )

    axis.set_title("Feature weight history")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weight")
    axis.grid(True, alpha=0.3)

    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        ncol=2,
        fontsize="small",
    )

    figure.tight_layout()

    output_path = run_dir / "weight_history.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return output_path


def plot_weight_change(
    run_dir: Path,
    weights_records: list[dict[str, Any]],
) -> Path | None:
    """
    初期エポックの重みに対する変化量をグラフ化する。
    """
    if not weights_records:
        return None

    initial_weights = weights_records[0]["weights"]

    epochs = [
        int(record["epoch"])
        for record in weights_records
    ]

    figure, axis = plt.subplots(figsize=(14, 8))

    for feature_name in FEATURE_NAMES:
        initial_value = float(
            initial_weights.get(feature_name, 0.0)
        )

        changes = [
            float(record["weights"].get(feature_name, 0.0))
            - initial_value
            for record in weights_records
        ]

        axis.plot(
            epochs,
            changes,
            label=feature_name,
            linewidth=1.0,
        )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_title("Feature weight change from initial epoch")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weight change")
    axis.grid(True, alpha=0.3)

    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        ncol=2,
        fontsize="small",
    )

    figure.tight_layout()

    output_path = run_dir / "weight_change.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training metrics and feature weights."
    )

    parser.add_argument(
        "run_dir",
        type=Path,
        help="Training run directory",
    )

    args = parser.parse_args()
    run_dir = args.run_dir

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory does not exist: {run_dir}"
        )

    metrics_path = run_dir / "metrics.jsonl"
    weights_path = run_dir / "weights.jsonl"

    metrics = (
        load_jsonl(metrics_path)
        if metrics_path.exists()
        else []
    )

    weights_records = (
        load_jsonl(weights_path)
        if weights_path.exists()
        else []
    )

    config = _load_config(run_dir)

    print(f"Run directory: {run_dir}")

    if config:
        print(f"Config: {config}")

    generated_files: list[Path] = []

    loss_path = plot_loss_curve(run_dir, metrics)
    if loss_path is not None:
        generated_files.append(loss_path)

    history_path = plot_weight_history(
        run_dir,
        weights_records,
    )
    if history_path is not None:
        generated_files.append(history_path)

    change_path = plot_weight_change(
        run_dir,
        weights_records,
    )
    if change_path is not None:
        generated_files.append(change_path)

    if not generated_files:
        print("No plottable records were found.")
        return

    print("Generated files:")

    for output_path in generated_files:
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
