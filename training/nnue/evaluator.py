from __future__ import annotations

import torch

from training.nnue.network import NNUE


class NNUEEvaluator:

    def __init__(
        self,
        model: NNUE,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = torch.device(device)

        self.model.eval()

    @torch.no_grad()
    def evaluate(
        self,
        accumulator_black: torch.Tensor,
        accumulator_white: torch.Tensor,
    ) -> float:

        accumulator_black = accumulator_black.to(
            self.device
        )

        accumulator_white = accumulator_white.to(
            self.device
        )

        value = self.model(
            accumulator_black,
            accumulator_white,
        )

        return float(value.item())
