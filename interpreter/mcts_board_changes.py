"""
interpreter/mcts_board_changes.py

MCTSの各variationに含まれる後続手について、

    後続手
      ↓
    局面変化
      ↓
    変更された升
      ↓
    利き・可動性・重要升制御などの変化

を具体的に抽出する。

このモジュールでは、指し手の「意図」を直接推定しない。

あくまで、

    「その手の前後で何が変化したか」

を記録する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cshogi

from features.activity import (
    all_attacks,
    important_squares,
)
from features.common import (
    iter_pieces,
)


# ============================================================
# Piece names
# ============================================================

_PIECE_NAMES = {
    cshogi.PAWN: "歩",
    cshogi.LANCE: "香",
    cshogi.KNIGHT: "桂",
    cshogi.SILVER: "銀",
    cshogi.BISHOP: "角",
    cshogi.ROOK: "飛",
    cshogi.GOLD: "金",
    cshogi.KING: "玉",
}


# ============================================================
# Basic data structures
# ============================================================


@dataclass(frozen=True)
class SquareChange:
    """
    1つの升で発生した駒配置の変化。
    """

    square: str
    before_piece: str | None
    after_piece: str | None


@dataclass(frozen=True)
class ControlChange:
    """
    ある色の利きが変化した升。
    """

    color: int
    added_squares: tuple[str, ...]
    removed_squares: tuple[str, ...]


@dataclass(frozen=True)
class MobilityChange:
    """
    合法手数の変化。
    """

    color: int
    before: int
    after: int
    change: int


@dataclass(frozen=True)
class ImportantControlChange:
    """
    重要升に対する利きの変化。
    """

    color: int
    before: int
    after: int
    change: int


@dataclass(frozen=True)
class DefenseControlChange:
    """
    自玉周辺の守備範囲に対する利きの変化。
    """

    color: int
    before: int
    after: int
    change: int


@dataclass(frozen=True)
class BoardChange:
    """
    1つの後続手によって観測された盤面変化。
    """

    move_usi: str
    piece_name: str
    from_square: str | None
    to_square: str

    changed_squares: tuple[SquareChange, ...]

    black_control: ControlChange
    white_control: ControlChange

    mobility: tuple[MobilityChange, ...]

    important_control: tuple[
        ImportantControlChange,
        ...
    ]

    defense_control: tuple[
        DefenseControlChange,
        ...
    ]


# ============================================================
# Square utilities
# ============================================================


def _square_to_usi(square: int) -> str:
    """
    cshogi 1.0.4 の内部square番号をUSI座標へ変換する。

    実測した対応:

        square 6  -> 1g
        square 15 -> 2g
        square 60 -> 7g
        square 59 -> 7f
        square 78 -> 9g

    cshogiでは、

        file = square // 9 + 1
        rank = square % 9

    の対応になっている。
    """

    file_number = (square // 9) + 1

    rank = chr(
        ord("a") + (square % 9)
    )

    return f"{file_number}{rank}"


def _piece_name(board: cshogi.Board, square: int) -> str | None:
    """
    指定升にある駒の名称を取得する。
    """

    piece = board.piece(square)

    if piece == 0:
        return None

    piece_type = cshogi.piece_to_piece_type(piece)

    return _PIECE_NAMES.get(
        piece_type,
        "駒",
    )


# ============================================================
# Board comparison
# ============================================================


def extract_changed_squares(
    before: cshogi.Board,
    after: cshogi.Board,
) -> tuple[SquareChange, ...]:
    """
    2局面を比較し、駒配置が変化した升を取得する。
    """

    changes: list[SquareChange] = []

    for square in range(81):
        before_piece = before.piece(square)
        after_piece = after.piece(square)

        if before_piece == after_piece:
            continue

        changes.append(
            SquareChange(
                square=_square_to_usi(square),
                before_piece=_piece_name(
                    before,
                    square,
                ),
                after_piece=_piece_name(
                    after,
                    square,
                ),
            )
        )

    return tuple(changes)


# ============================================================
# Move information
# ============================================================


def describe_move(
    board: cshogi.Board,
    move: int,
) -> tuple[str, str | None, str]:
    """
    moveから、

        USI手
        移動元
        移動先

    を取得する。
    """

    move_usi = cshogi.move_to_usi(move)

    to_square = _square_to_usi(
        cshogi.move_to(move)
    )

    if cshogi.move_is_drop(move):
        hand_piece_type = (
            cshogi.move_from_piece_type(move)
        )

        piece_name = _PIECE_NAMES.get(
            hand_piece_type,
            "駒",
        )

        return (
            move_usi,
            None,
            to_square,
        )

    from_square_index = cshogi.move_from(move)

    from_square = _square_to_usi(
        from_square_index
    )

    return (
        move_usi,
        from_square,
        to_square,
    )


def find_move_between_positions(
    before: cshogi.Board,
    after: cshogi.Board,
) -> int | None:
    """
    beforeからafterへ1手進んだときの指し手を特定する。
    """

    for move in before.legal_moves:
        candidate = before.copy()
        candidate.push(move)

        if candidate.pieces == after.pieces:
            if (
                candidate.pieces_in_hand
                == after.pieces_in_hand
            ):
                return move

    return None


# ============================================================
# Control changes
# ============================================================


def _controlled_squares(
    board: cshogi.Board,
    color: int,
) -> set[int]:
    """
    指定色が利かせている升の集合。
    """

    attacks = all_attacks(
        board,
        color,
    )

    result: set[int] = set()

    for squares in attacks.values():
        result.update(squares)

    return result


def extract_control_change(
    before: cshogi.Board,
    after: cshogi.Board,
    color: int,
) -> ControlChange:
    """
    指定色の利きの変化を取得する。
    """

    before_squares = _controlled_squares(
        before,
        color,
    )

    after_squares = _controlled_squares(
        after,
        color,
    )

    added = sorted(
        after_squares - before_squares
    )

    removed = sorted(
        before_squares - after_squares
    )

    return ControlChange(
        color=color,
        added_squares=tuple(
            _square_to_usi(square)
            for square in added
        ),
        removed_squares=tuple(
            _square_to_usi(square)
            for square in removed
        ),
    )


# ============================================================
# Mobility
# ============================================================


def _legal_move_count(
    board: cshogi.Board,
    color: int,
) -> int:
    """
    指定色の合法手数を取得する。
    """

    copied = board.copy()
    copied.turn = color

    return sum(
        1
        for _ in copied.legal_moves
    )


def extract_mobility_change(
    before: cshogi.Board,
    after: cshogi.Board,
    color: int,
) -> MobilityChange:
    """
    指定色の合法手数の変化を取得する。
    """

    before_count = _legal_move_count(
        before,
        color,
    )

    after_count = _legal_move_count(
        after,
        color,
    )

    return MobilityChange(
        color=color,
        before=before_count,
        after=after_count,
        change=after_count - before_count,
    )


# ============================================================
# Important-square control
# ============================================================


def _important_control_count(
    board: cshogi.Board,
    color: int,
) -> int:
    """
    重要升に対する利きの数を取得する。
    """

    squares = important_squares(
        board,
        color,
    )

    attacks = all_attacks(
        board,
        color,
    )

    return sum(
        len(attacked & squares)
        for attacked in attacks.values()
    )


def extract_important_control_change(
    before: cshogi.Board,
    after: cshogi.Board,
    color: int,
) -> ImportantControlChange:
    """
    F08相当の重要升への利きの変化を取得する。
    """

    before_count = _important_control_count(
        before,
        color,
    )

    after_count = _important_control_count(
        after,
        color,
    )

    return ImportantControlChange(
        color=color,
        before=before_count,
        after=after_count,
        change=after_count - before_count,
    )


# ============================================================
# Defense control
# ============================================================


def _king_defense_squares(
    board: cshogi.Board,
    color: int,
) -> set[int]:
    """
    自玉周辺の防御対象升を取得する。

    features.activity.pyのF17/F18と
    同じ考え方で、自玉とその周囲8升を対象とする。
    """

    king_square = board.king_square(color)

    file_index = king_square % 9
    rank_index = king_square // 9

    result = {
        king_square,
    }

    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue

            nf = file_index + df
            nr = rank_index + dr

            if 0 <= nf < 9 and 0 <= nr < 9:
                result.add(
                    nr * 9 + nf
                )

    return result


def _defense_control_count(
    board: cshogi.Board,
    color: int,
) -> int:
    """
    自玉周辺への自軍の利きの数を取得する。
    """

    defense_squares = _king_defense_squares(
        board,
        color,
    )

    attacks = all_attacks(
        board,
        color,
    )

    return sum(
        len(attacked & defense_squares)
        for attacked in attacks.values()
    )


def extract_defense_control_change(
    before: cshogi.Board,
    after: cshogi.Board,
    color: int,
) -> DefenseControlChange:
    """
    自玉周辺への守備利きの変化を取得する。
    """

    before_count = _defense_control_count(
        before,
        color,
    )

    after_count = _defense_control_count(
        after,
        color,
    )

    return DefenseControlChange(
        color=color,
        before=before_count,
        after=after_count,
        change=after_count - before_count,
    )


# ============================================================
# Main analyzer
# ============================================================


def analyze_board_change(
    before: cshogi.Board,
    after: cshogi.Board,
) -> BoardChange:
    """
    before → after の1手分の具体的な変化を解析する。
    """

    move = find_move_between_positions(
        before,
        after,
    )

    if move is None:
        raise ValueError(
            "Could not identify the move between the two positions."
        )

    move_usi, from_square, to_square = describe_move(
        before,
        move,
    )

    if from_square is not None:
        from_square_index = cshogi.move_from(move)

        piece_name = _piece_name(
            before,
            from_square_index,
        ) or "駒"
    else:
        piece_type = cshogi.move_from_piece_type(
            move
        )

        piece_name = _PIECE_NAMES.get(
            piece_type,
            "駒",
        )

    changed_squares = extract_changed_squares(
        before,
        after,
    )

    black_control = extract_control_change(
        before,
        after,
        cshogi.BLACK,
    )

    white_control = extract_control_change(
        before,
        after,
        cshogi.WHITE,
    )

    mobility = (
        extract_mobility_change(
            before,
            after,
            cshogi.BLACK,
        ),
        extract_mobility_change(
            before,
            after,
            cshogi.WHITE,
        ),
    )

    important_control = (
        extract_important_control_change(
            before,
            after,
            cshogi.BLACK,
        ),
        extract_important_control_change(
            before,
            after,
            cshogi.WHITE,
        ),
    )

    defense_control = (
        extract_defense_control_change(
            before,
            after,
            cshogi.BLACK,
        ),
        extract_defense_control_change(
            before,
            after,
            cshogi.WHITE,
        ),
    )

    return BoardChange(
        move_usi=move_usi,
        piece_name=piece_name,
        from_square=from_square,
        to_square=to_square,
        changed_squares=changed_squares,
        black_control=black_control,
        white_control=white_control,
        mobility=mobility,
        important_control=important_control,
        defense_control=defense_control,
    )

# ============================================================
# MCTS後続手による個別の利き変化
# ============================================================


@dataclass(frozen=True)
class PieceControlChange:
    """
    1つの駒について、before / after で変化した利きを表す。

    piece_square:
        利きを発生させている駒の升

    piece_name:
        駒名

    added_squares:
        afterで新しく利くようになった升

    removed_squares:
        afterで利かなくなった升
    """

    piece_square: str
    piece_name: str
    added_squares: tuple[str, ...]
    removed_squares: tuple[str, ...]


@dataclass(frozen=True)
class LocalControlChange:
    """
    盤面全体ではなく、駒ごとの利きの変化。
    """

    color: int
    changes: tuple[PieceControlChange, ...]


def _piece_control_map(
    board: cshogi.Board,
    color: int,
) -> dict[int, set[int]]:
    """
    指定色の各駒について、

        駒の内部square番号
        ->
        その駒が利いているsquare番号の集合

    を返す。
    """

    return {
        square: set(attacks)
        for square, attacks in all_attacks(
            board,
            color,
        ).items()
    }


def extract_piece_control_change(
    before: cshogi.Board,
    after: cshogi.Board,
    color: int,
) -> LocalControlChange:
    """
    before / after 間で、指定色の各駒の利きが
    どのように変化したかを取得する。

    駒自身が移動した場合も、

        beforeの駒位置
        afterの駒位置

    を別の駒として扱うのではなく、
    同じ駒の移動による盤面変化としてまとめる。

    現段階では駒の同一性を完全に追跡せず、
    square単位でbefore / afterを比較する。
    """

    before_map = _piece_control_map(
        before,
        color,
    )

    after_map = _piece_control_map(
        after,
        color,
    )

    changes: list[PieceControlChange] = []

    all_piece_squares = (
        set(before_map.keys())
        | set(after_map.keys())
    )

    for square in sorted(all_piece_squares):
        before_attacks = before_map.get(
            square,
            set(),
        )

        after_attacks = after_map.get(
            square,
            set(),
        )

        added = after_attacks - before_attacks
        removed = before_attacks - after_attacks

        if not added and not removed:
            continue

        piece = after.piece(square)

        if piece == 0:
            piece = before.piece(square)

        piece_type = (
            cshogi.piece_to_piece_type(piece)
        )

        piece_name = _PIECE_NAMES.get(
            piece_type,
            "駒",
        )

        changes.append(
            PieceControlChange(
                piece_square=_square_to_usi(
                    square
                ),
                piece_name=piece_name,
                added_squares=tuple(
                    sorted(
                        _square_to_usi(
                            item
                        )
                        for item in added
                    )
                ),
                removed_squares=tuple(
                    sorted(
                        _square_to_usi(
                            item
                        )
                        for item in removed
                    )
                ),
            )
        )

    return LocalControlChange(
        color=color,
        changes=tuple(changes),
    )


def extract_local_control_changes(
    before: cshogi.Board,
    after: cshogi.Board,
) -> tuple[
    LocalControlChange,
    LocalControlChange,
]:
    """
    黒・白双方について、駒ごとの利きの変化を取得する。
    """

    black = extract_piece_control_change(
        before,
        after,
        cshogi.BLACK,
    )

    white = extract_piece_control_change(
        before,
        after,
        cshogi.WHITE,
    )

    return black, white


# ============================================================
# MCTS variation全体の盤面変化
# ============================================================


@dataclass(frozen=True)
class VariationBoardChange:
    """
    MCTSの1つのvariationについて、
    各手の盤面変化を保持する。
    """

    move_index: int
    move_usi: str
    before: cshogi.Board
    after: cshogi.Board
    board_change: BoardChange
    black_control: LocalControlChange
    white_control: LocalControlChange


def analyze_variation_board_changes(
    positions: Sequence[cshogi.Board],
) -> tuple[VariationBoardChange, ...]:
    """
    MCTS variationのS0...Svから、
    各遷移の盤面変化を解析する。

    positions[0] = S0
    positions[1] = S1
    ...
    positions[n] = Sn

    各 S(i) -> S(i+1) について、
    指し手・盤面変化・局所的な利き変化を取得する。
    """

    results: list[VariationBoardChange] = []

    for index in range(
        len(positions) - 1
    ):
        before = positions[index]
        after = positions[index + 1]

        move = find_move_between_positions(
            before,
            after,
        )

        if move is None:
            continue

        move_usi = cshogi.move_to_usi(move)

        board_change = analyze_board_change(
            before,
            after,
        )

        black_control, white_control = (
            extract_local_control_changes(
                before,
                after,
            )
        )

        results.append(
            VariationBoardChange(
                move_index=index,
                move_usi=move_usi,
                before=before,
                after=after,
                board_change=board_change,
                black_control=black_control,
                white_control=white_control,
            )
        )

    return tuple(results)
