"""
features/weakness.py

F30:
    弱点形成・解消

盤面上に存在する弱点を評価し、

    ・弱点がどの程度存在するか
    ・相手からどの程度集中攻撃を受けているか
    ・自軍による守備がどの程度不足しているか
    ・相手駒の侵入対象になっているか

を連続値として表現する。

F30は弱点の「良し悪し」そのものではなく、
盤面上の弱点構造を表す。

指し手の解釈では、

    ΔF30 = F30(after) - F30(before)

を利用し、

    ΔF30 > 0
        → 弱点形成・拡大

    ΔF30 < 0
        → 弱点解消

として扱う。
"""

from __future__ import annotations

from dataclasses import dataclass

from .activity import (
    BLACK,
    WHITE,
    all_attacks,
    controlled_squares,
)
from .common import (
    board_pieces,
    piece_color,
    square_distance,
    tanh_normalize,
)


# ----------------------------------------------------------------------
# 重み
# ----------------------------------------------------------------------

DEFENSE_THINNESS_WEIGHT = 1.0
ATTACK_CONCENTRATION_WEIGHT = 1.5
INVASION_RISK_WEIGHT = 1.0

NORMALIZATION_SCALE = 10.0


@dataclass(frozen=True)
class F30Weakness:
    """F30の弱点評価."""

    score: float
    defense_thinness: float
    attack_concentration: float
    invasion_risk: float


@dataclass(frozen=True)
class F30WeaknessChange:
    """F30の局面変化."""

    black: float
    white: float
    difference: float


def _opponent(color: int) -> int:
    """相手側の色を返す。"""

    if color == BLACK:
        return WHITE

    return BLACK


def _candidate_squares(
    board,
    color: int,
) -> set[int]:
    """
    弱点評価対象となるマスを取得する。

    以下を対象とする。

        ・自軍駒が存在するマス
        ・相手駒が攻撃しているマス

    これにより、盤上の意味の薄い空きマスを
    無制限に評価することを避ける。
    """

    opponent = _opponent(color)

    squares = {
        piece.square
        for piece in board_pieces(
            board,
            color,
        )
    }

    opponent_attacks = all_attacks(
        board,
        opponent,
    )

    for targets in opponent_attacks.values():
        squares.update(targets)

    return squares


def defense_thinness_score(
    board,
    color: int,
) -> float:
    """
    自軍の守備が薄い度合いを計算する。

    評価対象マスについて、

        自軍の利きが少ない
        → 弱点

    とする。

    0に近いほど守備が厚く、
    1に近いほど守備が薄い。
    """

    squares = _candidate_squares(
        board,
        color,
    )

    if not squares:
        return 0.0

    own_attacks = all_attacks(
        board,
        color,
    )

    attack_counts = {
        square: 0
        for square in squares
    }

    for targets in own_attacks.values():
        for target in targets:
            if target in attack_counts:
                attack_counts[target] += 1

    total_weakness = 0.0

    for count in attack_counts.values():

        # 守備が0なら最大の弱点
        # 守備が1なら中程度
        # 2以上なら弱点として小さく評価
        if count == 0:
            total_weakness += 1.0

        elif count == 1:
            total_weakness += 0.5

    return total_weakness / float(
        len(squares)
    )


def attack_concentration_score(
    board,
    color: int,
) -> float:
    """
    相手からの集中攻撃の度合いを計算する。

    同一マスに対して相手の利きが
    複数集中するほど値を大きくする。

    0～1程度の値を返す。
    """

    opponent = _opponent(color)

    attacks = all_attacks(
        board,
        opponent,
    )

    if not attacks:
        return 0.0

    control_counts: dict[int, int] = {}

    for targets in attacks.values():

        for target in targets:

            control_counts[target] = (
                control_counts.get(
                    target,
                    0,
                )
                + 1
            )

    if not control_counts:
        return 0.0

    total = sum(
        control_counts.values()
    )

    if total == 0:
        return 0.0

    concentrated = sum(
        count
        for count in control_counts.values()
        if count >= 2
    )

    return float(concentrated) / float(total)


def _pawn_defended_squares(
    board,
    color: int,
) -> set[int]:
    """
    自軍の歩によって守られているマスを取得する。

    現在のcshogiで利用している
    board_pieces / piece情報のみを利用する。
    """

    result: set[int] = set()

    attacks = all_attacks(
        board,
        color,
    )

    for piece in board_pieces(
        board,
        color,
    ):

        # base_type 1 = Pawn
        if piece.base_type != 1:
            continue

        result.update(
            attacks.get(
                piece.square,
                set(),
            )
        )

    return result


def pawn_defense_lack_score(
    board,
    color: int,
) -> float:
    """
    歩による守備が不足している度合いを計算する。

    相手から攻撃されているマスのうち、
    歩によって守られていないマスを評価する。
    """

    opponent = _opponent(color)

    opponent_attacks = all_attacks(
        board,
        opponent,
    )

    attacked_squares: set[int] = set()

    for targets in opponent_attacks.values():
        attacked_squares.update(
            targets
        )

    if not attacked_squares:
        return 0.0

    pawn_defense = _pawn_defended_squares(
        board,
        color,
    )

    lack_count = sum(
        1
        for square in attacked_squares
        if square not in pawn_defense
    )

    return float(lack_count) / float(
        len(attacked_squares)
    )


def invasion_risk_score(
    board,
    color: int,
) -> float:
    """
    相手駒の侵入リスクを計算する。

    相手駒が自陣側のマスへ到達可能な場合、
    侵入リスクとして評価する。

    現段階では、相手駒の現在の利きを利用した
    保守的な近似値とする。
    """

    opponent = _opponent(color)

    opponent_attacks = all_attacks(
        board,
        opponent,
    )

    if not opponent_attacks:
        return 0.0

    risk = 0
    total = 0

    for piece in board_pieces(
        board,
        opponent,
    ):

        targets = opponent_attacks.get(
            piece.square,
            set(),
        )

        for target in targets:

            total += 1

            # 相手駒から自軍側への侵入候補を
            # 距離によって評価する。
            own_pieces = board_pieces(
                board,
                color,
            )

            if not own_pieces:
                continue

            nearest = min(
                square_distance(
                    target,
                    own_piece.square,
                )
                for own_piece in own_pieces
            )

            if nearest <= 2.0:
                risk += 1

    if total == 0:
        return 0.0

    return float(risk) / float(total)


def f30_weakness(
    board,
    color: int,
) -> F30Weakness:
    """
    F30弱点評価を計算する。
    """

    defense_thinness = (
        defense_thinness_score(
            board,
            color,
        )
    )

    attack_concentration = (
        attack_concentration_score(
            board,
            color,
        )
    )


    invasion_risk = (
        invasion_risk_score(
            board,
            color,
        )
    )

    raw_score = (
        DEFENSE_THINNESS_WEIGHT
        * defense_thinness
        + ATTACK_CONCENTRATION_WEIGHT
        * attack_concentration
        + INVASION_RISK_WEIGHT
        * invasion_risk
    )

    score = tanh_normalize(
        raw_score,
        NORMALIZATION_SCALE,
    )

    return F30Weakness(
        score=score,
        defense_thinness=defense_thinness,
        attack_concentration=attack_concentration,
        invasion_risk=invasion_risk,
    )


def f30_weakness_change(
    before,
    after,
) -> F30WeaknessChange:
    """
    指し手によるF30の変化量を計算する。
    """

    black_before = f30_weakness(
        before,
        BLACK,
    )

    black_after = f30_weakness(
        after,
        BLACK,
    )

    white_before = f30_weakness(
        before,
        WHITE,
    )

    white_after = f30_weakness(
        after,
        WHITE,
    )

    black_change = (
        black_after.score
        - black_before.score
    )

    white_change = (
        white_after.score
        - white_before.score
    )

    return F30WeaknessChange(
        black=black_change,
        white=white_change,
        difference=(
            black_change
            - white_change
        ),
    )