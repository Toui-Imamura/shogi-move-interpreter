"""
transition.py

将棋の局面遷移を扱う共通モジュール。

研究上の基本的な考え方:

    S0
     |
     | move
     v
    S1
     |
     | move
     v
    S2
     |
    ...
     |
     v
    Sv

特徴量の「指し手による変化」を計算するために使用する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import cshogi


@dataclass(frozen=True)
class PositionTransition:
    """
    1手による局面遷移。

    Attributes
    ----------
    before:
        指し手を指す前の局面 S0。
    move:
        cshogiのエンコードされた指し手。
    after:
        指し手を指した後の局面 S1。
    """

    before: cshogi.Board
    move: int
    after: cshogi.Board


@dataclass(frozen=True)
class Variation:
    """
    1本の指し手変化列。

    Attributes
    ----------
    initial:
        初期局面 S0。
    moves:
        適用した指し手列。
    positions:
        S0を含む各局面。
        positions[0] = S0
        positions[1] = S1
        ...
        positions[-1] = Sv
    """

    initial: cshogi.Board
    moves: tuple[int, ...]
    positions: tuple[cshogi.Board, ...]

    @property
    def final(self) -> cshogi.Board:
        """最終局面 Svを返す。"""
        return self.positions[-1]

    @property
    def length(self) -> int:
        """適用した指し手数を返す。"""
        return len(self.moves)


def copy_board(board: cshogi.Board) -> cshogi.Board:
    """
    局面をコピーする。

    元のBoardを変更しないため、特徴量計算では
    原則としてこの関数を利用する。
    """
    return board.copy()


def apply_move(
    board: cshogi.Board,
    move: int,
) -> cshogi.Board:
    """
    1手を適用して新しい局面を返す。

    Parameters
    ----------
    board:
        指し手適用前の局面。
    move:
        cshogiでエンコードされた合法手。

    Returns
    -------
    cshogi.Board
        指し手適用後の局面。

    Raises
    ------
    ValueError
        指し手が合法でない場合。
    """
    next_board = board.copy()

    if not next_board.is_legal(move):
        raise ValueError(f"Illegal move: {move}")

    next_board.push(move)

    return next_board


def apply_usi_move(
    board: cshogi.Board,
    usi: str,
) -> cshogi.Board:
    """
    USI形式の指し手を適用して新しい局面を返す。

    例:
        "7g7f"
        "2b3c+"
        "P*7f"
    """
    next_board = board.copy()

    move = next_board.move_from_usi(usi)

    if not next_board.is_legal(move):
        raise ValueError(f"Illegal USI move: {usi}")

    next_board.push(move)

    return next_board


def make_transition(
    board: cshogi.Board,
    move: int,
) -> PositionTransition:
    """
    1手分の局面遷移を作成する。

    元のboardは変更しない。
    """
    before = board.copy()
    after = apply_move(before, move)

    return PositionTransition(
        before=before,
        move=move,
        after=after,
    )


def make_variation(
    board: cshogi.Board,
    moves: Sequence[int],
) -> Variation:
    """
    指し手列を適用して1本の変化列を作成する。

    Parameters
    ----------
    board:
        初期局面 S0。
    moves:
        適用する指し手列。

    Returns
    -------
    Variation
        S0からSvまでの局面列。

    Notes
    -----
    元のboardは変更しない。
    """
    initial = board.copy()
    current = initial.copy()

    positions: List[cshogi.Board] = [initial]
    applied_moves: List[int] = []

    for move in moves:
        current = apply_move(current, move)

        positions.append(current)
        applied_moves.append(move)

    return Variation(
        initial=initial,
        moves=tuple(applied_moves),
        positions=tuple(positions),
    )


def make_usi_variation(
    board: cshogi.Board,
    usi_moves: Iterable[str],
) -> Variation:
    """
    USI形式の指し手列からVariationを作成する。
    """
    initial = board.copy()
    current = initial.copy()

    positions: List[cshogi.Board] = [initial]
    applied_moves: List[int] = []

    for usi in usi_moves:
        move = current.move_from_usi(usi)

        if not current.is_legal(move):
            raise ValueError(f"Illegal USI move: {usi}")

        current.push(move)

        positions.append(current.copy())
        applied_moves.append(move)

    return Variation(
        initial=initial,
        moves=tuple(applied_moves),
        positions=tuple(positions),
    )


def transition_from_usi(
    board: cshogi.Board,
    usi: str,
) -> PositionTransition:
    """
    USI形式の1手からPositionTransitionを作成する。
    """
    move = board.move_from_usi(usi)

    return make_transition(board, move)


def position_at(
    variation: Variation,
    ply: int,
) -> cshogi.Board:
    """
    Variation内の指定手数の局面を取得する。

    ply=0 ならS0。
    ply=1 ならS1。
    ...
    """
    if ply < 0 or ply >= len(variation.positions):
        raise IndexError(
            f"ply out of range: {ply}, "
            f"valid range is 0..{len(variation.positions) - 1}"
        )

    return variation.positions[ply]


def final_position(
    variation: Variation,
) -> cshogi.Board:
    """Variationの最終局面Svを取得する。"""
    return variation.final
