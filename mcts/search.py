from __future__ import annotations

import math
from dataclasses import dataclass

import cshogi
import torch

from mcts.node import MCTSNode
from training.nnue.accumulator import evaluate_board


@dataclass
class MCTSVariation:
    """
    MCTSによって得られた1本の変化手順。
    """

    moves: list[int]
    visits: int
    value: float


class SimpleMCTS:
    """
    NNUE評価を利用する簡易MCTS。

    注意:
        現段階では研究用の最小実装であり、
        AlphaZeroのPolicy Networkや
        本格的なNNUE差分更新は使用しない。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        simulations: int = 100,
        max_depth: int = 5,
        exploration_constant: float = 1.4,
        device: str = "cpu",
    ):
        if simulations <= 0:
            raise ValueError(
                "simulations must be positive"
            )

        if max_depth <= 0:
            raise ValueError(
                "max_depth must be positive"
            )

        if exploration_constant < 0:
            raise ValueError(
                "exploration_constant must be non-negative"
            )

        self.model = model
        self.simulations = int(simulations)
        self.max_depth = int(max_depth)
        self.exploration_constant = float(
            exploration_constant
        )
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    def evaluate(self, board: cshogi.Board) -> float:
        """
        局面をNNUEで評価する。

        現在のNNUEはBlack視点の評価値を
        直接学習する設計であるため、
        ここでは評価値をそのまま返す。
        """

        return evaluate_board(
            board,
            self.model,
        )

    def legal_moves(
        self,
        board: cshogi.Board,
    ) -> list[int]:
        """
        合法手をリスト化する。
        """

        return list(board.legal_moves)

    def make_child(
        self,
        node: MCTSNode,
        move: int,
    ) -> MCTSNode:
        """
        指し手から子ノードを生成する。
        """

        child_board = node.board.copy()
        child_board.push(move)

        child_value = self.evaluate(child_board)

        return MCTSNode(
            board=child_board,
            parent=node,
            move=move,
            prior_value=child_value,
        )

    def expand(
        self,
        node: MCTSNode,
    ) -> None:
        """
        ノードの子ノードを生成する。
        """

        if node.children:
            return

        legal_moves = self.legal_moves(node.board)

        for move in legal_moves:
            node.children[move] = self.make_child(
                node,
                move,
            )

    def select_child(
        self,
        node: MCTSNode,
    ) -> MCTSNode:
        """
        UCB1によって子ノードを選択する。
        """

        if not node.children:
            raise ValueError(
                "cannot select child from an unexpanded node"
            )

        parent_visits = max(node.visits, 1)

        def score(child: MCTSNode) -> float:
            exploitation = child.mean_value

            exploration = (
                self.exploration_constant
                * math.sqrt(
                    math.log(parent_visits + 1.0)
                    / (child.visits + 1.0)
                )
            )

            return exploitation + exploration

        return max(
            node.children.values(),
            key=score,
        )

    def select(
        self,
        root: MCTSNode,
    ) -> tuple[MCTSNode, list[MCTSNode]]:
        """
        根から葉まで選択する。

        戻り値:
            leaf:
                到達した葉ノード

            path:
                根から葉までのノード列
        """

        node = root
        path = [node]
        depth = 0

        while depth < self.max_depth:
            if not node.children:
                break

            node = self.select_child(node)
            path.append(node)
            depth += 1

        return node, path

    def backup(
        self,
        path: list[MCTSNode],
        value: float,
    ) -> None:
        """
        評価値を経路上のノードへ逆伝播する。

        評価値はBlack視点で統一する。
        White手番のノードでは符号を反転する。
        """

        for node in reversed(path):
            node_value = value

            if node.board.turn == cshogi.WHITE:
                node_value = -value

            node.update(node_value)

    def run_simulation(
        self,
        root: MCTSNode,
    ) -> None:
        """
        MCTSを1回実行する。
        """

        leaf, path = self.select(root)

        if leaf.board.legal_moves:
            self.expand(leaf)

        value = self.evaluate(leaf.board)

        self.backup(
            path,
            value,
        )

    def search(
        self,
        board: cshogi.Board,
    ) -> MCTSNode:
        """
        指定局面からMCTSを実行する。

        戻り値:
            探索木のルートノード
        """

        if board is None:
            raise ValueError(
                "board must not be None"
            )

        root = MCTSNode(
            board=board.copy(),
            prior_value=self.evaluate(board),
        )

        self.expand(root)

        for _ in range(self.simulations):
            self.run_simulation(root)

        return root

    def principal_variation(
        self,
        root: MCTSNode,
        max_depth: int | None = None,
    ) -> list[int]:
        """
        各局面で訪問回数が最大の手を選び、
        1本の代表変化手順を抽出する。
        """

        if max_depth is None:
            max_depth = self.max_depth

        moves: list[int] = []
        node = root

        for _ in range(max_depth):
            if not node.children:
                break

            child = max(
                node.children.values(),
                key=lambda item: item.visits,
            )

            if child.move is None:
                break

            moves.append(child.move)
            node = child

        return moves

    def top_variations(
        self,
        root: MCTSNode,
        num_variations: int = 5,
        max_depth: int | None = None,
    ) -> list[MCTSVariation]:
        """
        ルート直下の訪問回数上位手から、
        複数の変化手順を生成する。

        現段階では、各候補手を起点に
        その後は訪問回数最大の手を選ぶ。
        """

        if num_variations <= 0:
            raise ValueError(
                "num_variations must be positive"
            )

        if max_depth is None:
            max_depth = self.max_depth

        children = sorted(
            root.children.values(),
            key=lambda child: child.visits,
            reverse=True,
        )

        variations: list[MCTSVariation] = []

        for child in children[:num_variations]:
            if child.move is None:
                continue

            moves = [child.move]
            node = child

            for _ in range(max_depth - 1):
                if not node.children:
                    break

                next_node = max(
                    node.children.values(),
                    key=lambda item: item.visits,
                )

                if next_node.move is None:
                    break

                moves.append(next_node.move)
                node = next_node

            variations.append(
                MCTSVariation(
                    moves=moves,
                    visits=child.visits,
                    value=child.mean_value,
                )
            )

        return variations
