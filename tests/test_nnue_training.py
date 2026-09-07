from __future__ import annotations

import torch

from training.nnue.dataset import (
    NNUEPositionDataset,
)
from training.nnue.features import (
    num_feature_ids,
)
from training.nnue.network import (
    NNUE,
)
from training.nnue.trainer import (
    NNUETrainer,
)


def test_train_dataset():

    dataset = NNUEPositionDataset(
        "data/processed/positions.jsonl"
    )

    model = NNUE(
        num_features=num_feature_ids(),
        accumulator_size=256,
        hidden_size=32,
    )

    trainer = NNUETrainer(
        model,
        learning_rate=1e-3,
    )

    losses = trainer.train_dataset(
        dataset,
        batch_size=2,
        epochs=2,
        shuffle=False,
    )

    assert len(losses) > 0

    assert all(
        torch.isfinite(
            torch.tensor(loss)
        )
        for loss in losses
    )


def test_save(tmp_path):

    model = NNUE(
        num_features=num_feature_ids(),
        accumulator_size=256,
        hidden_size=32,
    )

    trainer = NNUETrainer(
        model,
    )

    path = tmp_path / "nnue.pt"

    trainer.save(path)

    assert path.exists()
