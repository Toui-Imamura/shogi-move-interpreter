from __future__ import annotations

import argparse
import math
import statistics

import torch
from torch.utils.data import DataLoader

from training.nnue.dataset import (
    NNUEPositionDataset,
    teacher_score_to_target,
)
from training.nnue.features import num_feature_ids
from training.nnue.network import NNUE
from training.nnue.trainer import NNUETrainer
from training.nnue.train import split_by_game


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/processed/teacher_values_wcsc36_295games.jsonl",
    )
    parser.add_argument(
        "--model",
        default="data/models/nnue/nnue_best.pt",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    dataset = NNUEPositionDataset(
        args.data,
        exclude_mate=True,
    )
    _, validation_dataset = split_by_game(
        dataset,
        validation_ratio=0.2,
        seed=args.seed,
    )

    model = NNUE(
        num_features=num_feature_ids(),
        accumulator_size=256,
        hidden_size=32,
    )

    state = torch.load(
        args.model,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)

    trainer = NNUETrainer(model, device=args.device)

    loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=trainer._collate,
    )

    predictions = []
    targets = []

    model.eval()
    with torch.no_grad():
        for black, white, target in loader:
            prediction = model(
                black.to(trainer.device),
                white.to(trainer.device),
            )
            predictions.extend(prediction.cpu().tolist())
            targets.extend(target.tolist())

    mse = statistics.mean(
        (p - t) ** 2 for p, t in zip(predictions, targets)
    )
    mae = statistics.mean(
        abs(p - t) for p, t in zip(predictions, targets)
    )

    mean_p = statistics.mean(predictions)
    mean_t = statistics.mean(targets)

    covariance = statistics.mean(
        (p - mean_p) * (t - mean_t)
        for p, t in zip(predictions, targets)
    )
    std_p = math.sqrt(
        statistics.mean((p - mean_p) ** 2 for p in predictions)
    )
    std_t = math.sqrt(
        statistics.mean((t - mean_t) ** 2 for t in targets)
    )
    correlation = (
        covariance / (std_p * std_t)
        if std_p > 0 and std_t > 0
        else float("nan")
    )

    print("Validation positions:", len(targets))
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Pearson correlation: {correlation:.6f}")
    print(f"Prediction mean/std: {mean_p:.6f} / {std_p:.6f}")
    print(f"Target mean/std: {mean_t:.6f} / {std_t:.6f}")

    print("\nPrediction ranges:")
    for low, high in [
        (-1.0, -0.5),
        (-0.5, 0.0),
        (0.0, 0.5),
        (0.5, 1.0),
    ]:
        count = sum(
            low <= p < high
            for p in predictions
        )
        print(f"[{low: .1f}, {high: .1f}): {count}")


if __name__ == "__main__":
    main()
