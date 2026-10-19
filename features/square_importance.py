"""81マスごとの利き情報と重要度を計算する。"""

from __future__ import annotations

from dataclasses import dataclass

import cshogi

from features.activity import all_attacks, important_squares
from features.common import BLACK, WHITE


def square_to_usi(square: int) -> str:
    """盤面インデックスをUSI座標に変換する。"""
    if not 0 <= square < 81:
        raise ValueError(f"invalid square: {square}")

    return f"{square // 9 + 1}{chr(ord('a') + square % 9)}"


@dataclass(frozen=True)
class SquareEffect:
    """1マスについて、両陣営の利きと重要度を記録する。"""

    square: int
    usi: str
    piece: int
    black_attackers: int
    white_attackers: int
    black_controls: bool
    white_controls: bool
    contested: bool
    importance: float


def build_square_effects(
    board: cshogi.Board,
) -> tuple[SquareEffect, ...]:
    """
    全81マスについて利きの情報を作成する。

    空きマスも対象とする。
    利きの数は、各駒の利き先集合に含まれる回数で数える。
    """
    black_attacks = all_attacks(board, BLACK)
    white_attacks = all_attacks(board, WHITE)

    black_counts = [0] * 81
    white_counts = [0] * 81

    for targets in black_attacks.values():
        for square in targets:
            black_counts[square] += 1

    for targets in white_attacks.values():
        for square in targets:
            white_counts[square] += 1

    black_important = important_squares(board, BLACK)
    white_important = important_squares(board, WHITE)
    important_union = black_important | white_important

    results = []

    for square in range(81):
        black_count = black_counts[square]
        white_count = white_counts[square]

        # 初期版の透明なヒューリスティック。
        # 利きの集中と両陣営の競合、既存の重要地点を評価する。
        contested = black_count > 0 and white_count > 0
        importance = (
            black_count
            + white_count
            + 2.0 * min(black_count, white_count)
            + (2.0 if square in important_union else 0.0)
        )

        results.append(
            SquareEffect(
                square=square,
                usi=square_to_usi(square),
                piece=board.piece(square),
                black_attackers=black_count,
                white_attackers=white_count,
                black_controls=black_count > 0,
                white_controls=white_count > 0,
                contested=contested,
                importance=importance,
            )
        )

    return tuple(results)


def changed_square_effects(
    before: cshogi.Board,
    after: cshogi.Board,
) -> tuple[tuple[SquareEffect, SquareEffect], ...]:
    """
    指し手の前後で利き情報が変わったマスを返す。

    戻り値は (変更前, 変更後) の組。
    駒配置の変化だけでなく、空きマスの利きの変化も対象とする。
    """
    before_effects = build_square_effects(before)
    after_effects = build_square_effects(after)

    changes = []

    for old, new in zip(before_effects, after_effects):
        if (
            old.piece != new.piece
            or old.black_attackers != new.black_attackers
            or old.white_attackers != new.white_attackers
            or old.importance != new.importance
        ):
            changes.append((old, new))

    return tuple(changes)


def top_important_squares(
    board: cshogi.Board,
    *,
    top_n: int = 10,
    include_occupied: bool = True,
) -> tuple[SquareEffect, ...]:
    """重要度の高いマスを順に返す。空きマスも候補に含める。"""
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    effects = build_square_effects(board)

    if not include_occupied:
        effects = tuple(item for item in effects if item.piece == cshogi.NONE)

    return tuple(
        sorted(
            effects,
            key=lambda item: (
                item.importance,
                item.contested,
                item.black_attackers + item.white_attackers,
                -item.square,
            ),
            reverse=True,
        )[:top_n]
    )
