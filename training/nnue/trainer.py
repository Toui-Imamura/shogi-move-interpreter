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
            self.model.parameters(),
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

        accumulator_black = accumulator_black.to(self.device)
        accumulator_white = accumulator_white.to(self.device)
        target = target.to(self.device)

        self.optimizer.zero_grad()

        prediction = self.model(
            accumulator_black,
            accumulator_white,
        )
        loss = self.loss_fn(prediction, target)

        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def _make_loader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
        )

    def train_dataset(
        self,
        dataset,
        batch_size: int = 8,
        epochs: int = 1,
        shuffle: bool = True,
    ) -> list[float]:
        """
        既存APIとの互換性を維持する。
        戻り値は従来どおり各バッチの損失。
        """

        loader = self._make_loader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        losses: list[float] = []

        for _ in range(epochs):
            for black, white, target in loader:
                losses.append(
                    self.train_step(black, white, target)
                )

        return losses

    def train_epoch(
        self,
        dataset,
        batch_size: int = 8,
    ) -> float:
        """1エポック分の平均学習損失を返す。"""

        loader = self._make_loader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        total_loss = 0.0
        total_samples = 0

        for black, white, target in loader:
            batch_size_actual = int(target.shape[0])
            loss = self.train_step(black, white, target)

            total_loss += loss * batch_size_actual
            total_samples += batch_size_actual

        if total_samples == 0:
            raise ValueError("Training dataset is empty")

        return total_loss / total_samples

    def evaluate_dataset(
        self,
        dataset,
        batch_size: int = 8,
    ) -> float:
        """勾配更新を行わず、平均検証損失を返す。"""

        if len(dataset) == 0:
            raise ValueError("Validation dataset is empty")

        loader = self._make_loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        try:
            with torch.no_grad():
                for black, white, target in loader:
                    black = black.to(self.device)
                    white = white.to(self.device)
                    target = target.to(self.device)

                    prediction = self.model(black, white)
                    loss = self.loss_fn(prediction, target)

                    batch_size_actual = int(target.shape[0])
                    total_loss += (
                        float(loss.item()) * batch_size_actual
                    )
                    total_samples += batch_size_actual
        finally:
            self.model.train(was_training)

        return total_loss / total_samples

    def _collate(self, batch: list[dict]):
        black_accumulators = []
        white_accumulators = []
        targets = []

        embedding = self.model.embedding

        for sample in batch:
            board = cshogi.Board(sfen=sample["sfen"])

            black, white = board_to_accumulators(
                board,
                embedding,
            )

            black_accumulators.append(black)
            white_accumulators.append(white)
            targets.append(sample["target"])

        return (
            torch.stack(black_accumulators),
            torch.stack(white_accumulators),
            torch.stack(targets),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path)
