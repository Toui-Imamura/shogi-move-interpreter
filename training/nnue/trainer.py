from __future__ import annotations

from pathlib import Path

import cshogi
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.nnue.accumulator import board_to_accumulators


class NNUETrainer:

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.device = torch.device(device)

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

        self.loss_fn = nn.MSELoss()

    def train_step(
        self,
        accumulator_black: torch.Tensor,
        accumulator_white: torch.Tensor,
        target: torch.Tensor,
    ) -> float:

        self.model.train()

        accumulator_black = accumulator_black.to(
            self.device
        )

        accumulator_white = accumulator_white.to(
            self.device
        )

        target = target.to(
            self.device
        )

        self.optimizer.zero_grad()

        prediction = self.model(
            accumulator_black,
            accumulator_white,
        )

        loss = self.loss_fn(
            prediction,
            target,
        )

        loss.backward()

        self.optimizer.step()

        return float(loss.item())

    def train_dataset(
        self,
        dataset,
        batch_size: int = 8,
        epochs: int = 1,
        shuffle: bool = True,
    ) -> list[float]:
        """
        NNUEPositionDatasetを使用して学習する。

        現段階では正しさを優先して、
        各SFENからAccumulatorを生成する。
        """

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
        )

        losses: list[float] = []

        for epoch in range(epochs):

            for (
                accumulator_black,
                accumulator_white,
                target,
            ) in loader:

                loss = self.train_step(
                    accumulator_black,
                    accumulator_white,
                    target,
                )

                losses.append(loss)

        return losses

    def _collate(
        self,
        batch: list[dict],
    ):
        """
        DatasetのSFENからAccumulatorを作る。
        """

        black_accumulators = []
        white_accumulators = []
        targets = []

        # embeddingはmodel自身が持っている
        embedding = self.model.embedding

        for sample in batch:

            board = cshogi.Board(
                sfen=sample["sfen"]
            )

            black, white = board_to_accumulators(
                board,
                embedding,
            )

            black_accumulators.append(
                black
            )

            white_accumulators.append(
                white
            )

            targets.append(
                sample["target"]
            )

        return (
            torch.stack(
                black_accumulators
            ),
            torch.stack(
                white_accumulators
            ),
            torch.stack(
                targets
            ),
        )

    def save(
        self,
        path: str | Path,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            self.model.state_dict(),
            path,
        )
