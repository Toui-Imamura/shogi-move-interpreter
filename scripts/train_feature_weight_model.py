from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

# scripts/から直接実行した場合でも、
# プロジェクトルートをimport対象にする。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.feature_weight_dataset import (  # noqa: E402
    FEATURE_NAMES,
    load_training_records,
    records_to_arrays,
)

from training.experiment_logging import (  # noqa: E402
    save_metrics_history,
    save_validation_predictions,
    save_weight_history,
)

from training.linear_feature_weight_model import (  # noqa: E402
    apply_standardization,
    clip_targets,
    evaluate_predictions,
    fit_standardization,
    train_linear_regression,
)


# 現時点で実際に計算されている特徴量。
# F13, F15, F34～F39は現段階では未実装のため除外する。
AVAILABLE_FEATURE_NAMES = (
    "F01",
    "F02",
    "F03",
    "F04",
    "F05",
    "F06",
    "F07",
    "F08",
    "F09",
    "F10",
    "F11",
    "F12",
    "F14",
    "F16",
    "F17",
    "F18",
    "F19",
    "F20",
    "F21",
    "F22",
    "F23",
    "F24",
    "F25",
    "F26",
    "F27",
    "F28",
    "F29",
    "F30",
    "F31",
    "F32",
    "F33",
    "F40",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a linear model for feature-weight estimation."
        )
    )

    parser.add_argument(
        "--teacher",
        type=Path,
        default=Path(
            "data/processed/"
            "teacher_transitions_wcsc36_10games.jsonl"
        ),
        help="Teacher transition JSONL path",
    )

    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "data/processed/"
            "feature_transitions_wcsc36.jsonl"
        ),
        help="Feature transition JSONL path",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/training_logs/"
            "linear_feature_weight_wcsc36_10games"
        ),
        help="Training output directory",
    )

    parser.add_argument(
        "--valid-games",
        type=int,
        default=2,
        help="Number of game IDs assigned to validation",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Gradient descent learning rate",
    )

    parser.add_argument(
        "--l2-strength",
        type=float,
        default=0.001,
        help="L2 regularization strength",
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Epoch logging interval",
    )

    parser.add_argument(
        "--target-mode",
        choices=("raw", "clipped_2000"),
        default="raw",
        help="Target preprocessing mode",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        file.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def split_indices_by_game(
    records: list[Any],
    valid_games: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """
    game_id単位でtrain/validationを分割する。
    """

    if valid_games <= 0:
        raise ValueError("valid_games must be positive")

    game_ids = sorted(
        {
            int(record.game_id)
            for record in records
        }
    )

    if len(game_ids) <= valid_games:
        raise ValueError(
            "Validation games must be smaller than total games: "
            f"total={len(game_ids)}, valid={valid_games}"
        )

    rng = np.random.default_rng(seed)
    shuffled_game_ids = list(game_ids)
    rng.shuffle(shuffled_game_ids)

    valid_game_ids = sorted(
        shuffled_game_ids[:valid_games]
    )
    train_game_ids = sorted(
        shuffled_game_ids[valid_games:]
    )

    train_game_set = set(train_game_ids)
    valid_game_set = set(valid_game_ids)

    train_indices = np.array(
        [
            index
            for index, record in enumerate(records)
            if int(record.game_id) in train_game_set
        ],
        dtype=np.int64,
    )

    valid_indices = np.array(
        [
            index
            for index, record in enumerate(records)
            if int(record.game_id) in valid_game_set
        ],
        dtype=np.int64,
    )

    return (
        train_indices,
        valid_indices,
        train_game_ids,
        valid_game_ids,
    )


def select_features(
    X: np.ndarray,
    feature_names: tuple[str, ...],
    selected_names: tuple[str, ...],
) -> np.ndarray:
    """
    全特徴量配列から学習対象特徴量だけを抽出する。
    """

    name_to_index = {
        name: index
        for index, name in enumerate(feature_names)
    }

    missing = [
        name
        for name in selected_names
        if name not in name_to_index
    ]

    if missing:
        raise ValueError(
            f"Selected features are not present: {missing}"
        )

    indices = [
        name_to_index[name]
        for name in selected_names
    ]

    return X[:, indices]


def make_metrics_row(
    *,
    epoch: int,
    train_metrics: dict[str, Any],
    valid_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "train": train_metrics,
        "valid": valid_metrics,
    }


def main() -> None:
    args = parse_args()

    if not args.teacher.exists():
        raise FileNotFoundError(
            f"Teacher file not found: {args.teacher}"
        )

    if not args.features.exists():
        raise FileNotFoundError(
            f"Feature file not found: {args.features}"
        )

    print("=== Loading dataset ===")

    records = load_training_records(
        teacher_path=args.teacher,
        feature_path=args.features,
    )

    X_all, y_all = records_to_arrays(records)

    print(f"record_count: {len(records)}")
    print(f"X shape: {X_all.shape}")
    print(f"y shape: {y_all.shape}")

    X = select_features(
        X_all,
        tuple(FEATURE_NAMES),
        AVAILABLE_FEATURE_NAMES,
    )

    y = np.asarray(y_all, dtype=np.float64)

    if args.target_mode == "raw":
        target_clip_limit = None
    elif args.target_mode == "clipped_2000":
        target_clip_limit = 2000.0
    else:
        raise ValueError(
            f"Unsupported target mode: {args.target_mode}"
        )

    y = clip_targets(
        y,
        target_clip_limit,
    )

    (
        train_indices,
        valid_indices,
        train_game_ids,
        valid_game_ids,
    ) = split_indices_by_game(
        records,
        valid_games=args.valid_games,
        seed=args.seed,
    )

    X_train_raw = X[train_indices]
    y_train = y[train_indices]

    X_valid_raw = X[valid_indices]
    y_valid = y[valid_indices]

    # 標準化統計量は学習データだけから計算する。
    standardization_stats = fit_standardization(
        X_train_raw
    )

    X_train = apply_standardization(
        X_train_raw,
        standardization_stats,
    )

    X_valid = apply_standardization(
        X_valid_raw,
        standardization_stats,
    )

    print()
    print("=== Split ===")
    print(f"train games: {train_game_ids}")
    print(f"valid games: {valid_game_ids}")
    print(f"train records: {len(train_indices)}")
    print(f"valid records: {len(valid_indices)}")

    print()
    print("=== Target mode ===")
    print(f"mode: {args.target_mode}")
    print(f"clip limit: {target_clip_limit}")

    print()
    print("=== Training ===")

    model, history = train_linear_regression(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        feature_names=AVAILABLE_FEATURE_NAMES,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2_strength=args.l2_strength,
        log_interval=args.log_interval,
        seed=args.seed,
    )

    train_predictions = model.predict(X_train)
    valid_predictions = model.predict(X_valid)

    train_metrics = evaluate_predictions(
        y_train,
        train_predictions,
    )

    valid_metrics = evaluate_predictions(
        y_valid,
        valid_predictions,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_type": "linear_regression",
        "target_mode": args.target_mode,
        "target_clip_limit": target_clip_limit,
        "teacher_path": str(args.teacher),
        "feature_path": str(args.features),
        "feature_names": list(AVAILABLE_FEATURE_NAMES),
        "feature_count": len(AVAILABLE_FEATURE_NAMES),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2_strength": args.l2_strength,
        "log_interval": args.log_interval,
        "seed": args.seed,
        "valid_games": args.valid_games,
        "train_game_ids": train_game_ids,
        "valid_game_ids": valid_game_ids,
        "train_record_count": int(len(train_indices)),
        "valid_record_count": int(len(valid_indices)),
    }

    write_json(
        output_dir / "config.json",
        config,
    )

    write_json(
        output_dir / "standardization.json",
        {
            "feature_names": list(AVAILABLE_FEATURE_NAMES),
            "mean": standardization_stats.mean.tolist(),
            "std": standardization_stats.std.tolist(),
        },
    )

    write_json(
        output_dir / "metrics_summary.json",
        {
            "train": train_metrics,
            "valid": valid_metrics,
        },
    )

    # 学習中の評価指標を履歴として保存する。
    # metrics.jsonlには、ログ対象epochごとの
    # train/validationのloss、MAE、RMSEを記録する。
    save_metrics_history(
        experiment_dir=output_dir,
        epochs=history.epochs,
        train_loss=history.train_loss,
        valid_loss=history.valid_loss,
        train_mae=history.train_mae,
        valid_mae=history.valid_mae,
        train_rmse=history.train_rmse,
        valid_rmse=history.valid_rmse,
    )

    # weight_history.jsonlには、学習中の各ログ時点における
    # 全特徴量の重みとbiasを保存する。
    save_weight_history(
        experiment_dir=output_dir,
        epochs=history.epochs,
        weight_history=history.weight_history,
        bias_history=history.bias_history,
    )

    # weights.jsonlは既存のグラフスクリプトとの互換性のため、
    # 最終重みを従来形式で保存する。
    weights_rows: list[dict[str, Any]] = []

    weights_rows.append(
        {
            "epoch": int(args.epochs),
            **{
                name: float(weight)
                for name, weight in zip(
                    AVAILABLE_FEATURE_NAMES,
                    model.weights,
                )
            },
            "bias": float(model.bias),
        }
    )

    write_jsonl(
        output_dir / "weights.jsonl",
        weights_rows,
    )

    write_json(
        output_dir / "final_model.json",
        {
            "feature_names": list(model.feature_names),
            "weights": model.weights.tolist(),
            "bias": float(model.bias),
        },
    )

    # 各検証レコードの予測値を保存する。
    prediction_rows: list[dict[str, Any]] = []

    for local_index, record_index in enumerate(valid_indices):
        record = records[int(record_index)]

        prediction_rows.append(
            {
                "game_id": int(record.game_id),
                "ply": int(record.ply),
                "sfen": record.sfen,
                "move": record.move,
                "target": float(y_valid[local_index]),
                "prediction": float(
                    valid_predictions[local_index]
                ),
                "absolute_error": float(
                    abs(
                        y_valid[local_index]
                        - valid_predictions[local_index]
                    )
                ),
            }
        )

    save_validation_predictions(
        experiment_dir=output_dir,
        rows=prediction_rows,
    )

    print()
    print("=== Final metrics ===")
    print("Train:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value}")

    print("Validation:")
    for key, value in valid_metrics.items():
        print(f"  {key}: {value}")

    print()
    print("=== Final weights ===")

    sorted_weights = sorted(
        zip(
            AVAILABLE_FEATURE_NAMES,
            model.weights,
        ),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )

    for name, weight in sorted_weights:
        print(f"{name}: {float(weight):.8f}")

    print()
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
