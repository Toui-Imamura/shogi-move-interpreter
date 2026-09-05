"""
defense.py

F23:
    防御力の変化

設計:
    Defense = a1 * Defenders
           + a2 * KingControl
           + a3 * Response

    Defenders:
        F17で定義した「玉周辺の守備駒数」

    KingControl:
        F18で定義した「玉周辺における自軍の利き」

    Response:
        相手攻撃への対応可能性を表す内部指標。
        将来的にF25として独立した特徴量にする。

注意:
    この特徴量は指し手の「意図」そのものではなく、
    局面における防御面の状態・変化を表す。
"""

from __future__ import annotations

from dataclasses import dataclass

import cshogi

from .activity import (
    BLACK,
    WHITE,
    controlled_squares,
    defensive_piece_count,
    defense_control,
    defense_area,
)
from .common import (
    king_square,
    tanh_normalize,
)


# ============================================================
# F23 パラメータ
# ============================================================

# 現段階では研究用の初期値。
# 最終的には棋譜データ等から調整する。
A1_DEFENDERS = 1.0
A2_KING_CONTROL = 1.0
A3_RESPONSE = 1.0


# Response内部の重み
RESPONSE_LEGAL_DEFENSE_WEIGHT = 1.0
RESPONSE_KING_ESCAPE_WEIGHT = 1.0
RESPONSE_COUNTER_ATTACK_WEIGHT = 1.0


# ============================================================
# Response
# ============================================================

@dataclass(frozen=True)
class ResponseComponents:
    """
    防御側の対応力を構成する3要素。

    LegalDefense:
        合法的に対応できる手の多さ

    KingEscape:
        玉の逃げ場の多さ

    CounterAttack:
        反撃可能性の代理指標
    """

    legal_defense: float
    king_escape: float
    counter_attack: float


def _king_escape_count(
    board: cshogi.Board,
    color: int,
) -> int:
    """
    指定した色の玉の合法的な移動先の数を計算する。

    board.turnに依存しないように、局面のコピーを作成して
    対象色を手番に設定した上で合法手を調べる。
    """

    king = king_square(board, color)

    if king is None:
        return 0

    temp = board.copy()
    temp.turn = color

    king_usi = cshogi.SQUARE_NAMES[king]

    count = 0

    for move in temp.legal_moves:
        usi = cshogi.move_to_usi(move)

        # 玉の移動元が現在の玉の位置と一致するものだけを数える。
        #
        # 例:
        #   5i4h
        #
        # の場合、usi[:2] == "5i"
        if len(usi) >= 4 and usi[:2] == king_usi:
            count += 1

    return count


def _legal_defense_score(
    board: cshogi.Board,
    color: int,
) -> float:
    """
    指定した色が持つ合法手数を計算する。

    board.turnに依存しないように、局面のコピーを作成して
    対象色を手番に設定した上で合法手を調べる。

    現段階では、その陣営が持つ合法手数を
    対応可能性の基本的な代理指標とする。

    将来的には、
    「相手の攻撃に直接対応できる合法手」
    に限定する。
    """

    temp = board.copy()
    temp.turn = color

    return float(len(list(temp.legal_moves)))


def _counter_attack_score(
    board: cshogi.Board,
    color: int,
) -> float:
    """
    CounterAttackを計算する。

    現段階では、相手玉周辺への自軍の利きを
    反撃可能性の代理指標として使用する。

    将来的にはMCTSによる反撃成立可能性へ拡張する。
    """

    opponent = WHITE if color == BLACK else BLACK

    opponent_king_area = defense_area(
        board,
        opponent,
    )

    own_control = controlled_squares(
        board,
        color,
    )

    return float(
        len(own_control.intersection(opponent_king_area))
    )


def response_components(
    board: cshogi.Board,
    color: int,
) -> ResponseComponents:
    """
    Responseの構成要素を取得する。
    """

    legal_defense = _legal_defense_score(
        board,
        color,
    )

    king_escape = float(
        _king_escape_count(
            board,
            color,
        )
    )

    counter_attack = _counter_attack_score(
        board,
        color,
    )

    return ResponseComponents(
        legal_defense=legal_defense,
        king_escape=king_escape,
        counter_attack=counter_attack,
    )


def response_score(
    board: cshogi.Board,
    color: int,
) -> float:
    """
    Responseの統合値を計算する。
    """

    components = response_components(
        board,
        color,
    )

    raw_score = (
        RESPONSE_LEGAL_DEFENSE_WEIGHT
        * components.legal_defense
        + RESPONSE_KING_ESCAPE_WEIGHT
        * components.king_escape
        + RESPONSE_COUNTER_ATTACK_WEIGHT
        * components.counter_attack
    )

    return tanh_normalize(
        raw_score,
        scale=20.0,
    )


# ============================================================
# F23 防御力
# ============================================================

@dataclass(frozen=True)
class F23Defense:
    """
    F23 防御力。

    defenders:
        玉周辺の守備駒数

    king_control:
        玉周辺における自軍の利き

    response:
        相手攻撃への対応力

    score:
        統合した防御力
    """

    defenders: float
    king_control: float
    response: float
    score: float


def f23_defense(
    board: cshogi.Board,
    color: int,
) -> F23Defense:
    """
    F23 防御力を計算する。
    """

    # --------------------------------------------------------
    # Defenders
    # F17の既存実装を利用
    # --------------------------------------------------------

    defenders = float(
        defensive_piece_count(
            board,
            color,
        )
    )

    # --------------------------------------------------------
    # KingControl
    # F18の既存定義と同じく、
    # 自玉周辺に対する自軍の利きを利用
    # --------------------------------------------------------

    king_control = float(
        defense_control(
            board,
            color,
        )
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    response = response_score(
        board,
        color,
    )

    # --------------------------------------------------------
    # 統合
    # --------------------------------------------------------

    raw_score = (
        A1_DEFENDERS * defenders
        + A2_KING_CONTROL * king_control
        + A3_RESPONSE * response
    )

    score = tanh_normalize(
        raw_score,
        scale=20.0,
    )

    return F23Defense(
        defenders=defenders,
        king_control=king_control,
        response=response,
        score=score,
    )


def f23_defense_change(
    before: cshogi.Board,
    after: cshogi.Board,
    color: int,
) -> F23Defense:
    """
    F23 防御力の変化を計算する。

    after - before
    """

    before_value = f23_defense(
        before,
        color,
    )

    after_value = f23_defense(
        after,
        color,
    )

    return F23Defense(
        defenders=(
            after_value.defenders
            - before_value.defenders
        ),
        king_control=(
            after_value.king_control
            - before_value.king_control
        ),
        response=(
            after_value.response
            - before_value.response
        ),
        score=(
            after_value.score
            - before_value.score
        ),
    )