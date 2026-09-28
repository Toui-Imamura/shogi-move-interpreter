from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from training.nnue.dataset import NNUEPositionDataset
from training.nnue.features import num_feature_ids
from training.nnue.network import NNUE
from training.nnue.trainer import NNUETrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the research NNUE model."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the NNUE training dataset.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the trained NNUE model.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--accumulator-size",
        type=int,
        default=256,
        help="NNUE accumulator size.",
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="Hidden layer size.",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Training device.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than 0.")

    if args.accumulator_size <= 0:
        raise ValueError("--accumulator-size must be greater than 0.")

    if args.hidden_size <= 0:
        raise ValueError("--hidden-size must be greater than 0.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False."
        )

    data_path = Path(args.data)
    output_path = Path(args.output)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training dataset was not found: {data_path}"
        )

    dataset = NNUEPositionDataset(data_path)

    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    num_features = num_feature_ids()

    model = NNUE(
        num_features=num_features,
        accumulator_size=args.accumulator_size,
        hidden_size=args.hidden_size,
    )

    trainer = NNUETrainer(
        model=model,
        learning_rate=args.learning_rate,
        device=args.device,
    )

    print("=== NNUE training ===")
    print(f"dataset: {data_path}")
    print(f"records: {len(dataset)}")
    print(f"features: {num_features}")
    print(f"accumulator size: {args.accumulator_size}")
    print(f"hidden size: {args.hidden_size}")
    print(f"batch size: {args.batch_size}")
    print(f"learning rate: {args.learning_rate}")
    print(f"epochs: {args.epochs}")
    print(f"device: {args.device}")

    losses = trainer.train_dataset(
        dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
    )

    if not losses:
        raise RuntimeError("No training losses were produced.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(output_path)

    print()
    print("=== Training finished ===")
    print(f"steps: {len(losses)}")
    print(f"initial loss: {losses[0]:.6f}")
    print(f"final loss: {losses[-1]:.6f}")
    print(f"model saved: {output_path}")


if __name__ == "__main__":
    main()
