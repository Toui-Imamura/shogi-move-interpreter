"""
response.py

F25:
    相手攻撃への対応

設計:

    Response =
        a1 * LegalDefense
      + a2 * KingEscape
      + a3 * CounterAttack

LegalDefense:
    相手から攻撃されている自駒に対する
    直接的な合法対応手数。

    1. 攻撃駒を取る対応
    2. 攻撃された駒を移動して回避する対応
    3. 攻撃線を遮断・防御して攻撃状態を解消する対応

KingEscape:
    自玉の合法的な逃走手数。

CounterAttack:
    相手玉周辺への反撃可能性。

注意:
    各値は「対応可能性」の代理指標であり、
    実際の最善応手の成功確率を意味しない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set

import cshogi

from .activity import (
    BLACK,
    WHITE,
    all_attacks,
    controlled_squares,
    defense_area,
    opponent,
    piece_color,
)
from .common import (
    tanh_normalize,
)


# ------------------------------------------------------------
# 重み
# ------------------------------------------------------------

A1_LEGAL_DEFENSE = 1.0
A2_KING_ESCAPE = 1.0
A3_COUNTER_ATTACK = 1.0

NORMALIZATION_SCALE = 20.0


# ------------------------------------------------------------
# データクラス
# ------------------------------------------------------------

@dataclass(frozen=True)
class F25Response:
    """
    F25の対応可能性。

    legal_defense:
        攻撃に対する直接的な対応手数。

    king_escape:
        玉の逃走手数。

    counter_attack:
        相手玉周辺への反撃可能性。

    score:
        3要素を統合した対応スコア。
    """

    legal_defense: float
    king_escape: float
    counter_attack: float
    score: float


@dataclass(frozen=True)
class F25ResponseChange:
    """
    F25の指し手前後の変化量。
    """

    black: float
    white: float
    difference: float


# ------------------------------------------------------------
# 攻撃対象の取得
# ------------------------------------------------------------

def attacked_piece_squares(
    board: cshogi.Board,
    color: int,
) -> Set[int]:
    """
    指定陣営の駒のうち、相手から攻撃されている駒の位置を返す。

    注意:
        activity.all_attacks() の攻撃情報を利用する。
        王手回避などの合法性はここでは扱わない。
    """

    opponent_color = opponent(color)

    opponent_attacks = all_attacks(
        board,
        opponent_color,
    )

    attacked: Set[int] = set()

    for target_square in range(81):
        piece = board.piece(target_square)

        if piece == 0:
            continue

        if piece_color(piece) != color:
            continue

        for attacks in opponent_attacks.values():
            if target_square in attacks:
                attacked.add(target_square)
                break

    return attacked


# ------------------------------------------------------------
# 攻撃駒の取得
# ------------------------------------------------------------

def attacking_piece_squares(
    board: cshogi.Board,
    target_square: int,
    opponent_color: int,
) -> Set[int]:
    """
    指定された自駒を攻撃している相手駒の位置を返す。
    """

    opponent_attacks = all_attacks(
        board,
        opponent_color,
    )

    return {
        attacker_square
        for attacker_square, attacks in opponent_attacks.items()
        if target_square in attacks
    }


# ------------------------------------------------------------
# LegalDefense
# ------------------------------------------------------------

def _legal_defense_score(
    board: cshogi.Board,
    color: int,
) -> float:
    """
    相手から攻撃されている駒に対する
    直接的な合法対応手数を計算する。

    対応として数えるもの:

        1. 攻撃駒を取る
        2. 攻撃された駒を逃がす
        3. 攻撃を遮断・防御して攻撃状態を解消する

    判定方法:

        各合法手について、その手を実行した後の局面を作成し、
        指定陣営の攻撃されている駒が減少するかを確認する。

    同一の合法手が複数の条件を満たす場合でも、
    1手として1回だけ数える。
    """

    attacked_before = attacked_piece_squares(
        board,
        color,
    )

    if not attacked_before:
        return 0.0

    temp = board.copy()
    temp.turn = color

    response_moves: Set[int] = set()

    for move in temp.legal_moves:
        next_board = temp.copy()
        next_board.push(move)

        attacked_after = attacked_piece_squares(
            next_board,
            color,
        )

        # 攻撃されていた自駒のうち、
        # 少なくとも1つへの攻撃状態が解消された場合、
        # その手を直接的な防御対応手として数える。
        if len(attacked_after.intersection(attacked_before)) < len(attacked_before):
            response_moves.add(move)

    return float(len(response_moves))


# ------------------------------------------------------------
# KingEscape
# ------------------------------------------------------------

def _king_escape_score(
    board: cshogi.Board,
    color: int,
) -> float:
    """
    自玉の合法的な逃走手数を数える。
    """

    king_square = board.king_square(color)

    if king_square is None:
        return 0.0

    temp = board.copy()
    temp.turn = color

    count = 0

    for move in temp.legal_moves:
        move_from = cshogi.move_from(move)

        if move_from == king_square:
            count += 1

    return float(count)


# ------------------------------------------------------------
# CounterAttack
# ------------------------------------------------------------

def _counter_attack_score(
    board: cshogi.Board,
    color: int,
) -> float:
    """
    相手玉周辺への反撃可能性を計算する。

    相手玉周辺8マスに対して、
    自陣営が持つ利きの数を代理指標とする。
    """

    opponent_color = opponent(color)

    opponent_king_area = defense_area(
        board,
        opponent_color,
    )

    own_control = controlled_squares(
        board,
        color,
    )

    return float(
        len(
            own_control.intersection(
                opponent_king_area,
            )
        )
    )


# ------------------------------------------------------------
# F25
# ------------------------------------------------------------

def f25_response(
    board: cshogi.Board,
    color: int,
) -> F25Response:
    """
    指定陣営のF25を計算する。
    """

    legal_defense = _legal_defense_score(
        board,
        color,
    )

    king_escape = _king_escape_score(
        board,
        color,
    )

    counter_attack = _counter_attack_score(
        board,
        color,
    )

    raw_score = (
        A1_LEGAL_DEFENSE * legal_defense
        + A2_KING_ESCAPE * king_escape
        + A3_COUNTER_ATTACK * counter_attack
    )

    score = tanh_normalize(
        raw_score,
        scale=NORMALIZATION_SCALE,
    )

    return F25Response(
        legal_defense=legal_defense,
        king_escape=king_escape,
        counter_attack=counter_attack,
        score=score,
    )


# ------------------------------------------------------------
# F25 change
# ------------------------------------------------------------

def f25_response_change(
    before: cshogi.Board,
    after: cshogi.Board,
    color: int,
) -> F25Response:
    """
    指し手前後におけるF25の変化量を計算する。
    """

    before_value = f25_response(
        before,
        color,
    )

    after_value = f25_response(
        after,
        color,
    )

    return F25Response(
        legal_defense=(
            after_value.legal_defense
            - before_value.legal_defense
        ),
        king_escape=(
            after_value.king_escape
            - before_value.king_escape
        ),
        counter_attack=(
            after_value.counter_attack
            - before_value.counter_attack
        ),
        score=(
            after_value.score
            - before_value.score
        ),
    )


def f25_response_change_both(
    before: cshogi.Board,
    after: cshogi.Board,
) -> F25ResponseChange:
    """
    Black / White双方のF25変化量を計算する。
    """

    black = f25_response_change(
        before,
        after,
        BLACK,
    )

    white = f25_response_change(
        before,
        after,
        WHITE,
    )

    return F25ResponseChange(
        black=black.score,
        white=white.score,
        difference=black.score - white.score,
    )
