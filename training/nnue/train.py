from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from torch.utils.data import Subset

from training.nnue.dataset import NNUEPositionDataset
from training.nnue.features import num_feature_ids
from training.nnue.network import NNUE
from training.nnue.trainer import NNUETrainer


def split_by_game(
    dataset: NNUEPositionDataset,
    validation_ratio: float,
    seed: int,
):
    """同一対局が学習・検証の両方に入らないように分割する。"""

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    game_indices: dict[str, list[int]] = {}

    for index, record in enumerate(dataset.records):
        game_indices.setdefault(record.game_id, []).append(index)

    game_ids = list(game_indices)
    if len(game_ids) < 2:
        raise ValueError(
            "At least two distinct games are required"
        )

    rng = random.Random(seed)
    rng.shuffle(game_ids)

    validation_game_count = round(
        len(game_ids) * validation_ratio
    )
    validation_game_count = max(
        1,
        min(len(game_ids) - 1, validation_game_count),
    )

    validation_games = set(game_ids[:validation_game_count])

    train_indices = []
    validation_indices = []

    for game_id, indices in game_indices.items():
        if game_id in validation_games:
            validation_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return Subset(dataset, train_indices), Subset(
        dataset,
        validation_indices,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train the research NNUE model"
    )
    parser.add_argument(
        "--data",
        default="data/processed/teacher_values_wcsc36_295games.jsonl",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        default="data/models/nnue",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = NNUEPositionDataset(
        args.data,
        exclude_mate=True,
    )

    if len(dataset) == 0:
        raise ValueError("No non-mate training records found")

    train_dataset, validation_dataset = split_by_game(
        dataset,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    print(f"All normal-score positions: {len(dataset)}")
    print(f"Training positions: {len(train_dataset)}")
    print(f"Validation positions: {len(validation_dataset)}")
    print(f"Device: {args.device}")

    model = NNUE(
        num_features=num_feature_ids(),
        accumulator_size=256,
        hidden_size=32,
    )

    trainer = NNUETrainer(
        model,
        learning_rate=args.learning_rate,
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = output_dir / "nnue_best.pt"
    history_path = output_dir / "training_history.csv"

    best_validation_loss = float("inf")

    with history_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "validation_loss",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss = trainer.train_epoch(
                train_dataset,
                batch_size=args.batch_size,
            )
            validation_loss = trainer.evaluate_dataset(
                validation_dataset,
                batch_size=args.batch_size,
            )

            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} "
                f"train_loss={train_loss:.6f} "
                f"validation_loss={validation_loss:.6f}"
            )

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.8f}",
                    "validation_loss": f"{validation_loss:.8f}",
                }
            )
            f.flush()

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                trainer.save(best_model_path)
                print(f"Saved best model: {best_model_path}")

    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
