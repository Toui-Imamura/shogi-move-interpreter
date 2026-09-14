"""
plot_training_log.py

学習ログから以下のグラフを作成する。

- loss_curve.png
- weight_history.png
- weight_change.png
- feature_weights/F01.png ～ feature_weights/F40.png
- feature_weights_grid.png

使用例:
    python scripts/plot_training_log.py data/training_logs/run_name
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.logging_utils import FEATURE_NAMES, load_jsonl


def load_json(path: Path) -> dict[str, Any]:
    """
    JSONファイルを読み込む。
    """
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return data


def ensure_output_directory(path: Path) -> None:
    """
    出力先ディレクトリを作成する。
    """
    path.mkdir(parents=True, exist_ok=True)


def extract_epochs(records: list[dict[str, Any]]) -> list[int]:
    """
    ログレコードからEpoch番号を抽出する。
    """
    epochs: list[int] = []

    for record in records:
        if "epoch" not in record:
            raise ValueError("Each log record must contain 'epoch'")

        epochs.append(int(record["epoch"]))

    return epochs


def plot_loss_curve(
    metrics_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    学習損失の推移を描画する。
    """
    if not metrics_records:
        return

    epochs = extract_epochs(metrics_records)

    train_loss = [
        float(record["train_loss"])
        for record in metrics_records
        if "train_loss" in record
    ]

    validation_loss = [
        float(record["valid_loss"])
        for record in metrics_records
        if "valid_loss" in record
    ]

    plt.figure(figsize=(10, 6))

    if train_loss:
        plt.plot(
            epochs[:len(train_loss)],
            train_loss,
            label="Train loss",
            linewidth=2,
        )

    if validation_loss:
        plt.plot(
            epochs[:len(validation_loss)],
            validation_loss,
            label="Validation loss",
            linewidth=2,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def extract_weight_records(
    weights_records: list[dict[str, Any]],
) -> tuple[list[int], dict[str, list[float]]]:
    """
    重みログを特徴量ごとの時系列へ変換する。

    Returns
    -------
    tuple[list[int], dict[str, list[float]]]
        Epoch一覧と、特徴量ごとの重み履歴
    """
    if not weights_records:
        return [], {feature_name: [] for feature_name in FEATURE_NAMES}

    epochs = extract_epochs(weights_records)

    histories: dict[str, list[float]] = {
        feature_name: []
        for feature_name in FEATURE_NAMES
    }

    for record in weights_records:
        weights = record.get("weights")

        if not isinstance(weights, dict):
            raise ValueError(
                "Each weight record must contain a 'weights' object"
            )

        for feature_name in FEATURE_NAMES:
            value = weights.get(feature_name, 0.0)
            histories[feature_name].append(float(value))

    return epochs, histories


def plot_weight_history(
    weights_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    全特徴量の重み履歴を1枚に描画する。

    これは従来の比較用グラフとして残す。
    """
    epochs, histories = extract_weight_records(weights_records)

    if not epochs:
        return

    plt.figure(figsize=(14, 8))

    for feature_name in FEATURE_NAMES:
        plt.plot(
            epochs,
            histories[feature_name],
            label=feature_name,
            linewidth=1,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Weight")
    plt.title("Feature weight history")
    plt.grid(True, alpha=0.3)
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        ncol=2,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_weight_change(
    weights_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    全特徴量について、初期Epochからの重みの変化量を描画する。
    """
    epochs, histories = extract_weight_records(weights_records)

    if not epochs:
        return

    plt.figure(figsize=(14, 8))

    for feature_name in FEATURE_NAMES:
        values = histories[feature_name]

        if not values:
            continue

        initial_value = values[0]
        changes = [
            value - initial_value
            for value in values
        ]

        plt.plot(
            epochs,
            changes,
            label=feature_name,
            linewidth=1,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Weight change from initial epoch")
    plt.title("Feature weight changes")
    plt.axhline(0.0, linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        ncol=2,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_individual_feature_weights(
    weights_records: list[dict[str, Any]],
    output_directory: Path,
) -> None:
    """
    特徴量ごとに重み履歴を個別画像として保存する。

    出力例:
        feature_weights/F01.png
        feature_weights/F02.png
        ...
        feature_weights/F40.png
    """
    epochs, histories = extract_weight_records(weights_records)

    if not epochs:
        return

    ensure_output_directory(output_directory)

    for feature_name in FEATURE_NAMES:
        values = histories[feature_name]

        if not values:
            continue

        final_weight = values[-1]
        initial_weight = values[0]
        weight_change = final_weight - initial_weight

        plt.figure(figsize=(9, 5.5))

        plt.plot(
            epochs,
            values,
            linewidth=2.2,
            marker="o",
            markersize=3,
        )

        plt.axhline(
            0.0,
            linewidth=1,
            linestyle="--",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Weight")
        plt.title(
            f"{feature_name} weight history\n"
            f"Initial: {initial_weight:.6f}  "
            f"Final: {final_weight:.6f}  "
            f"Change: {weight_change:+.6f}"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = output_directory / f"{feature_name}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()


def plot_individual_feature_weight_changes(
    weights_records: list[dict[str, Any]],
    output_directory: Path,
) -> None:
    """
    特徴量ごとに、初期Epochからの重みの変化量を個別画像として保存する。

    出力例:
        feature_weights/F01_change.png
        feature_weights/F02_change.png
        ...
    """
    epochs, histories = extract_weight_records(weights_records)

    if not epochs:
        return

    ensure_output_directory(output_directory)

    for feature_name in FEATURE_NAMES:
        values = histories[feature_name]

        if not values:
            continue

        initial_weight = values[0]
        changes = [
            value - initial_weight
            for value in values
        ]

        final_change = changes[-1]

        plt.figure(figsize=(9, 5.5))

        plt.plot(
            epochs,
            changes,
            linewidth=2.2,
            marker="o",
            markersize=3,
        )

        plt.axhline(
            0.0,
            linewidth=1,
            linestyle="--",
        )

        plt.xlabel("Epoch")
        plt.ylabel("Weight change")
        plt.title(
            f"{feature_name} weight change\n"
            f"Final change: {final_change:+.6f}"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = output_directory / f"{feature_name}_change.png"
        plt.savefig(output_path, dpi=150)
        plt.close()


def plot_feature_weights_grid(
    weights_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    F01～F40を一覧できるグリッド画像を作成する。

    4列×10行の構成。
    """
    epochs, histories = extract_weight_records(weights_records)

    if not epochs:
        return

    n_features = len(FEATURE_NAMES)
    n_columns = 4
    n_rows = (n_features + n_columns - 1) // n_columns

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(16, 24),
        squeeze=False,
    )

    axes_flat = axes.flatten()

    for index, feature_name in enumerate(FEATURE_NAMES):
        ax = axes_flat[index]
        values = histories[feature_name]

        if values:
            ax.plot(
                epochs,
                values,
                linewidth=1.5,
            )

            ax.axhline(
                0.0,
                linewidth=0.7,
                linestyle="--",
            )

            ax.set_title(
                f"{feature_name}\n"
                f"Final={values[-1]:.4f}",
                fontsize=9,
            )

        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Weight", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=7)

    # 余ったAxesを非表示にする
    for index in range(n_features, len(axes_flat)):
        axes_flat[index].axis("off")

    fig.suptitle(
        "Individual feature weight histories",
        fontsize=16,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training logs."
    )
    parser.add_argument(
        "run_directory",
        type=Path,
        help="Training log directory",
    )

    args = parser.parse_args()
    run_directory: Path = args.run_directory

    metrics_path = run_directory / "metrics.jsonl"
    weight_history_path = run_directory / "weight_history.jsonl"
    legacy_weights_path = run_directory / "weights.jsonl"

    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    # 学習途中の重み履歴を優先する。
    # weight_history.jsonlがない過去の実験結果では、
    # 従来のweights.jsonlを使用する。
    if weight_history_path.exists():
        weights_path = weight_history_path
        print(
            "Using weight history: "
            f"{weight_history_path}"
        )
    elif legacy_weights_path.exists():
        weights_path = legacy_weights_path
        print(
            "Using legacy final weights: "
            f"{legacy_weights_path}"
        )
    else:
        raise FileNotFoundError(
            "Neither weight_history.jsonl nor weights.jsonl "
            f"exists in {run_directory}"
        )

    metrics_records = load_jsonl(metrics_path)
    weights_records = load_jsonl(weights_path)

    plot_loss_curve(
        metrics_records,
        run_directory / "loss_curve.png",
    )

    plot_weight_history(
        weights_records,
        run_directory / "weight_history.png",
    )

    plot_weight_change(
        weights_records,
        run_directory / "weight_change.png",
    )

    individual_directory = run_directory / "feature_weights"

    plot_individual_feature_weights(
        weights_records,
        individual_directory,
    )

    plot_individual_feature_weight_changes(
        weights_records,
        individual_directory,
    )

    plot_feature_weights_grid(
        weights_records,
        run_directory / "feature_weights_grid.png",
    )

    print("Generated plots:")
    print(f"  {run_directory / 'loss_curve.png'}")
    print(f"  {run_directory / 'weight_history.png'}")
    print(f"  {run_directory / 'weight_change.png'}")
    print(f"  {individual_directory}")
    print(f"  {run_directory / 'feature_weights_grid.png'}")


if __name__ == "__main__":
    main()
