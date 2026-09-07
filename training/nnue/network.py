from __future__ import annotations

import torch
from torch import nn


class NNUE(nn.Module):
    """
    研究用の小型NNUE。

    sparse feature
        ↓
    feature embedding / accumulator
        ↓
    hidden layers
        ↓
    scalar evaluation
    """

    def __init__(
        self,
        num_features: int,
        accumulator_size: int = 256,
        hidden_size: int = 32,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_features,
            accumulator_size,
        )

        self.fc1 = nn.Linear(
            accumulator_size * 2,
            hidden_size,
        )

        self.fc2 = nn.Linear(
            hidden_size,
            hidden_size,
        )

        self.output = nn.Linear(
            hidden_size,
            1,
        )

        self.relu = nn.ReLU()

    def forward(
        self,
        accumulator_black: torch.Tensor,
        accumulator_white: torch.Tensor,
    ) -> torch.Tensor:

        x = torch.cat(
            [
                accumulator_black,
                accumulator_white,
            ],
            dim=-1,
        )

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return self.output(x).squeeze(-1)
