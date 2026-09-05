"""
king_safety.py

F26:
    玉の安全性

F24:
    玉の安全性の変化

設計:
    KingSafety =
        a1 * OwnControl
      - a2 * EnemyControl
      + a3 * Escape
      - a4 * Attackers

注意:
    F17「玉周辺の守備駒数」
    F18「玉周辺の利き」
    とは役割を分離する。

F26では、
    ・玉周辺に対する自軍の利き
    ・玉周辺に対する敵軍の利き
    ・玉の合法的な逃げ場所
    ・玉周辺を攻撃する敵駒数
を組み合わせて玉そのものの安全性を評価する。

この値は「良い手」「悪い手」や「指し手の意図」を直接表すものではない。
"""


from __future__ import annotations

from dataclasses import dataclass

import cshogi

from .activity import (
    BLACK,
    WHITE,
    all_attacks,
    defense_area,
)
from .common import (
    king_square,
    tanh_normalize,
)


# ============================================================
# 共通処理
# ============================================================


def opponent(color: int) -> int:
    """指定陣営の相手陣営を返す。"""
    return WHITE if color == BLACK else BLACK


def king_escape_count(
    board: cshogi.Board,
    color: int,
) -> int:
    """
    玉が合法的に移動できるマス数を数える。

    board.legal_movesは現在手番側の合法手しか返さないため、
    一時的に手番を対象陣営へ変更したコピー盤を使用する。

    現在のcshogi 1.0.4では、
    cshogi.move_to_usi() を利用して玉の移動を判定する。
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

        # USIの先頭2文字が移動元。
        # 玉以外の駒や駒打ちは対象外になる。
        if len(usi) >= 4 and usi[:2] == king_usi:
            count += 1

    return count


def own_king_control(
    board: cshogi.Board,
    color: int,
) -> int:
    """
    自軍が自玉周辺へ持つ利き数。

    同一マスへの複数の利きをそれぞれ数える。
    """

    area = defense_area(board, color)

    total = 0

    for attacks in all_attacks(board, color).values():
        total += len(attacks.intersection(area))

    return total


def enemy_king_control(
    board: cshogi.Board,
    color: int,
) -> int:
    """
    相手軍が自玉周辺へ持つ利き数。

    同一マスへの複数の利きをそれぞれ数える。
    """

    enemy = opponent(color)
    area = defense_area(board, color)

    total = 0

    for attacks in all_attacks(board, enemy).values():
        total += len(attacks.intersection(area))

    return total


def king_attackers(
    board: cshogi.Board,
    color: int,
) -> int:
    """
    自玉周辺を攻撃している相手駒数。

    「相手の駒が自玉周辺のどこかを攻撃している」
    場合、その駒を攻撃参加駒として1回だけ数える。
    """

    enemy = opponent(color)

    area = defense_area(board, color)

    attackers = set()

    for square, attacks in all_attacks(board, enemy).items():
        if attacks.intersection(area):
            attackers.add(square)

    return len(attackers)


# ============================================================
# F26 玉の安全性
# ============================================================


@dataclass(frozen=True)
class F26KingSafety:
    """
    F26 玉の安全性
    """

    black: float
    white: float
    difference: float

    black_own_control: int
    black_enemy_control: int
    black_escape: int
    black_attackers: int

    white_own_control: int
    white_enemy_control: int
    white_escape: int
    white_attackers: int


def _raw_king_safety(
    board: cshogi.Board,
    color: int,
) -> float:
    """
    玉の安全性の生値。

    KingSafety =
        a1 * OwnControl
      - a2 * EnemyControl
      + a3 * Escape
      - a4 * Attackers

    現段階では係数を暫定設定する。
    """

    own_control = own_king_control(
        board,
        color,
    )

    enemy_control = enemy_king_control(
        board,
        color,
    )

    escape = king_escape_count(
        board,
        color,
    )

    attackers = king_attackers(
        board,
        color,
    )

    # 暫定係数
    a1 = 1.0
    a2 = 1.0
    a3 = 1.0
    a4 = 2.0

    return (
        a1 * own_control
        - a2 * enemy_control
        + a3 * escape
        - a4 * attackers
    )


def f26_king_safety(
    board: cshogi.Board,
    normalization_c: float = 10.0,
) -> F26KingSafety:
    """
    F26 玉の安全性。

    正の値:
        玉周辺の防御状態が相対的に良い。

    負の値:
        玉周辺の攻撃圧力が相対的に強い。

    最終的には [-1, 1] に正規化する。
    """

    black_own = own_king_control(
        board,
        BLACK,
    )

    black_enemy = enemy_king_control(
        board,
        BLACK,
    )

    black_escape = king_escape_count(
        board,
        BLACK,
    )

    black_attackers = king_attackers(
        board,
        BLACK,
    )

    white_own = own_king_control(
        board,
        WHITE,
    )

    white_enemy = enemy_king_control(
        board,
        WHITE,
    )

    white_escape = king_escape_count(
        board,
        WHITE,
    )

    white_attackers = king_attackers(
        board,
        WHITE,
    )

    black_raw = _raw_king_safety(
        board,
        BLACK,
    )

    white_raw = _raw_king_safety(
        board,
        WHITE,
    )

    black = tanh_normalize(
        black_raw,
        normalization_c,
    )

    white = tanh_normalize(
        white_raw,
        normalization_c,
    )

    return F26KingSafety(
        black=black,
        white=white,
        difference=black - white,

        black_own_control=black_own,
        black_enemy_control=black_enemy,
        black_escape=black_escape,
        black_attackers=black_attackers,

        white_own_control=white_own,
        white_enemy_control=white_enemy,
        white_escape=white_escape,
        white_attackers=white_attackers,
    )


# ============================================================
# F24 玉の安全性の変化
# ============================================================


@dataclass(frozen=True)
class F24KingSafetyChange:
    """
    F24 玉の安全性の変化。

    F24 = ΔF26
    """

    black: float
    white: float
    difference: float


def f24_king_safety_change(
    before: cshogi.Board,
    after: cshogi.Board,
) -> F24KingSafetyChange:
    """
    F26について、

        ΔF26 = F26(after) - F26(before)

    を計算する。
    """

    before_value = f26_king_safety(
        before,
    )

    after_value = f26_king_safety(
        after,
    )

    black_change = (
        after_value.black
        - before_value.black
    )

    white_change = (
        after_value.white
        - before_value.white
    )

    return F24KingSafetyChange(
        black=black_change,
        white=white_change,
        difference=black_change - white_change,
    )


# ============================================================
# Combined extraction
# ============================================================


def extract_king_safety_features(
    board: cshogi.Board,
) -> dict[str, object]:
    """
    F26をまとめて抽出する。
    """

    return {
        "F26": f26_king_safety(board),
    }
