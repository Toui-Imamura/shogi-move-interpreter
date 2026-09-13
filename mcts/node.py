from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCTSNode:
    """
    簡易MCTSの探索木ノード。

    board:
        このノードが表す局面。

    parent:
        親ノード。

    move:
        親局面からこの局面へ到達するために指した手。

    prior_value:
        この局面を生成した時点でのNNUE評価値。

    visits:
        このノードが訪問された回数。

    total_value:
        訪問時に得られた評価値の累積。

    children:
        合法手ごとの子ノード。
    """

    board: object
    parent: Optional["MCTSNode"] = None
    move: Optional[int] = None
    prior_value: float = 0.0

    visits: int = 0
    total_value: float = 0.0

    children: dict[int, "MCTSNode"] = field(
        default_factory=dict
    )

    @property
    def mean_value(self) -> float:
        """
        訪問済みノードの平均評価値。
        """

        if self.visits == 0:
            return 0.0

        return self.total_value / self.visits

    def is_expanded(self) -> bool:
        """
        子ノードが生成済みか判定する。
        """

        return bool(self.children)

    def is_leaf(self) -> bool:
        """
        葉ノードか判定する。
        """

        return not self.children

    def update(self, value: float) -> None:
        """
        ノードの訪問回数と評価値を更新する。
        """

        self.visits += 1
        self.total_value += float(value)
