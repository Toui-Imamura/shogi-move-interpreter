from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import cshogi

from mcts.node import MCTSNode


@dataclass
class MCTSVariation:
    """
    MCTSによって得られた1本の変化手順。

    moves:
        ルート局面からの指し手列。

    visits:
        ルート直下の候補手が訪問された回数。

    value:
        その候補手ノードに保存されたBlack視点の平均評価値。
    """

    moves: list[int]
    visits: int
    value: float


class SimpleMCTS:
    """
    外部の局面評価器を利用する簡易MCTS。

    評価値の視点:
        Black視点に統一する。

        正の値:
            Black有利

        負の値:
            White有利

    evaluator:
        cshogi.Boardを受け取り、
        Black視点の数値評価値を返す関数。

        今後はYaneuraOuMCTSEvaluatorを使用する。

    注意:
        現段階では研究用の最小実装であり、
        AlphaZeroのPolicy Networkや
        本格的なNNUE差分更新は使用しない。
    """

    def __init__(
        self,
        evaluator: Callable[[cshogi.Board], float],
        simulations: int = 100,
        max_depth: int = 5,
        exploration_constant: float = 1.4,
    ):
        if evaluator is None:
            raise ValueError(
                "evaluator must not be None"
            )

        if not callable(evaluator):
            raise TypeError(
                "evaluator must be callable"
            )

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

        self.evaluator = evaluator
        self.simulations = int(simulations)
        self.max_depth = int(max_depth)
        self.exploration_constant = float(
            exploration_constant
        )

    def evaluate(
        self,
        board: cshogi.Board,
    ) -> float:
        """
        局面を外部評価器で評価する。

        評価値はBlack視点で統一する。
        """

        return float(
            self.evaluator(board)
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

        child_value = self.evaluate(
            child_board
        )

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

        legal_moves = self.legal_moves(
            node.board
        )

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

        評価値はBlack視点で統一しているため、

        - Black番:
            Blackに有利な子を選ぶ

        - White番:
            Whiteに有利な子、
            つまりBlack評価値が低い子を選ぶ

        とする。
        """

        if not node.children:
            raise ValueError(
                "cannot select child from an unexpanded node"
            )

        parent_visits = max(
            node.visits,
            1,
        )

        log_parent_visits = math.log(
            parent_visits + 1.0
        )

        def score(
            child: MCTSNode,
        ) -> float:
            if node.board.turn == cshogi.BLACK:
                exploitation = child.mean_value
            else:
                exploitation = -child.mean_value

            exploration = (
                self.exploration_constant
                * math.sqrt(
                    log_parent_visits
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

        そのため、各ノードには符号反転せず、
        同じBlack視点の評価値を加算する。
        """

        value = float(value)

        for node in reversed(path):
            node.update(value)

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

        value = self.evaluate(
            leaf.board
        )

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

        if root is None:
            raise ValueError(
                "root must not be None"
            )

        if max_depth is None:
            max_depth = self.max_depth

        if max_depth <= 0:
            return []

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

        各候補手を起点に、
        その後は訪問回数最大の手を選ぶ。

        visitsは、F37/F38の集約に利用する。
        """

        if root is None:
            raise ValueError(
                "root must not be None"
            )

        if num_variations <= 0:
            raise ValueError(
                "num_variations must be positive"
            )

        if max_depth is None:
            max_depth = self.max_depth

        if max_depth <= 0:
            return []

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
                    visits=int(child.visits),
                    value=float(child.mean_value),
                )
            )

        return variations
