"""攻撃対象・攻めの拠点・守備候補をマス単位で評価する。"""

from dataclasses import dataclass
from typing import List

import cshogi

from features.activity import all_attacks
from features.common import BLACK, WHITE, opponent, piece_color, piece_value
from features.square_importance import build_square_effects, square_to_usi


@dataclass(frozen=True)
class SquareCandidate:
    square: int
    usi: str
    score: float
    kind: str
    reasons: tuple[str, ...]


def _attack_counts(board, color: int) -> List[int]:
    """各マスに対する指定陣営の利きの数を数える。"""
    counts = [0] * 81
    for _from_square, targets in all_attacks(board, color).items():
        for target in targets:
            counts[target] += 1
    return counts


def _piece_value_at(board, square: int) -> float:
    piece = board.piece(square)
    if piece == cshogi.NONE:
        return 0.0
    return float(piece_value(board.piece_type(square)))


def analyze_square_strategy(
    board,
    color: int | None = None,
    top_n: int = 10,
) -> tuple[
    List[SquareCandidate],
    List[SquareCandidate],
    List[SquareCandidate],
]:
    """
    攻撃対象・攻めの拠点・守備候補を別々に抽出する。

    戻り値:
        (attack_targets, attack_bases, defenses)

    スコアは暫定ヒューリスティックであり、勝率ではない。
    """
    if color is None:
        color = board.turn
    if color not in (BLACK, WHITE):
        raise ValueError(f"invalid color: {color}")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    enemy = opponent(color)
    own_counts = _attack_counts(board, color)
    enemy_counts = _attack_counts(board, enemy)
    effects = {e.square: e for e in build_square_effects(board)}

    attack_targets: List[SquareCandidate] = []
    attack_bases: List[SquareCandidate] = []
    defenses: List[SquareCandidate] = []

    for square in range(81):
        own = own_counts[square]
        opposing = enemy_counts[square]
        piece = board.piece(square)
        owner = piece_color(piece) if piece != cshogi.NONE else None
        importance = float(effects[square].importance)
        usi = square_to_usi(square)

        # 1. 攻撃対象：自軍の利きがある敵駒のみ。
        if owner == enemy and own > 0:
            value = _piece_value_at(board, square)
            reasons = [
                f"自軍の利き{own}枚",
                f"対象駒の価値{value:g}",
                f"敵側の利き{opposing}枚",
            ]

            # 敵の守りを考慮する初期スコア。
            score = own * 1.5 + value * own
            score -= opposing * value * 0.6

            if own > opposing:
                reasons.append("利きの数で自軍が上回る")
            elif own == opposing:
                reasons.append("利きの数が同数")
            else:
                reasons.append("敵側の利きが上回る")

            attack_targets.append(
                SquareCandidate(
                    square, usi, score, "attack_target", tuple(reasons)
                )
            )

        # 2. 攻めの拠点：複数の自軍の利きが集まる空きマス。
        if piece == cshogi.NONE and own >= 2:
            reasons = [
                f"空きマスへの自軍の利き{own}枚",
                f"敵側の利き{opposing}枚",
                f"マス重要度{importance:g}",
            ]

            score = own * 1.5 + importance * 0.25

            # 両陣営が利きを持つ場合は競合地点として加点する。
            if opposing > 0:
                score += min(own, opposing) * 0.5
                reasons.append("両陣営の利きが競合")

            attack_bases.append(
                SquareCandidate(
                    square, usi, score, "attack_base", tuple(reasons)
                )
            )

        # 3a. 守備対象：敵の利きがある味方駒。
        if owner == color and opposing > 0:
            value = _piece_value_at(board, square)
            reasons = [
                f"味方駒の価値{value:g}",
                f"敵側の利き{opposing}枚",
                f"自軍の利き{own}枚",
            ]

            score = value * opposing
            score -= value * min(own, opposing) * 0.5
            score += importance * 0.25

            if own < opposing:
                reasons.append("自軍の利きが不足")
            elif own == opposing:
                reasons.append("利きの数が同数")
            else:
                reasons.append("自軍の利きが上回る")

            defenses.append(
                SquareCandidate(
                    square, usi, score, "defense_piece", tuple(reasons)
                )
            )

        # 3b. 守備候補地点：敵の利きがあり、自軍も利きを持つ空きマス。
        # 重要地点の守備候補を拾うための初期的な代理指標。
        if (
            piece == cshogi.NONE
            and opposing > 0
            and own > 0
            and opposing >= own
        ):
            reasons = [
                f"空きマスへの敵側の利き{opposing}枚",
                f"自軍の利き{own}枚",
                f"マス重要度{importance:g}",
                "競合地点の守備候補",
            ]
            score = importance + opposing - own * 0.5

            defenses.append(
                SquareCandidate(
                    square, usi, score, "defense_point", tuple(reasons)
                )
            )

    def sort_candidates(items):
        return sorted(items, key=lambda x: (-x.score, x.usi))[:top_n]

    return (
        sort_candidates(attack_targets),
        sort_candidates(attack_bases),
        sort_candidates(defenses),
    )


def analyze_square_candidates(
    board,
    color: int | None = None,
    top_n: int = 10,
) -> tuple[List[SquareCandidate], List[SquareCandidate]]:
    """
    既存呼び出しとの互換用。

    第1戻り値は攻撃対象と攻めの拠点を統合したリスト。
    新規コードでは analyze_square_strategy() を推奨する。
    """
    targets, bases, defenses = analyze_square_strategy(
        board, color=color, top_n=top_n
    )
    attacks = sorted(
        targets + bases,
        key=lambda x: (-x.score, x.usi),
    )[:top_n]
    return attacks, defenses


def print_square_candidates(
    board,
    color: int | None = None,
    top_n: int = 10,
) -> None:
    """3種類の候補と評価根拠を表示する。"""
    if color is None:
        color = board.turn

    targets, bases, defenses = analyze_square_strategy(
        board, color=color, top_n=top_n
    )
    side = "先手" if color == BLACK else "後手"

    print(f"\n対象：{side}")
    print("※ 評価値は暫定ヒューリスティックであり、勝率ではありません。")
    print("※ 利きの数は、駒を取れることや最善手であることを保証しません。")

    for title, candidates in (
        ("攻撃対象（敵駒）", targets),
        ("攻めの拠点（空きマス）", bases),
        ("守備候補（味方駒・競合地点）", defenses),
    ):
        print(f"\n【{title}】")
        if not candidates:
            print("該当する候補はありません。")
            continue

        for index, candidate in enumerate(candidates, start=1):
            print(
                f"{index:2}. {candidate.usi} "
                f"評価値={candidate.score:.2f}"
            )
            print("    根拠：" + "、".join(candidate.reasons))
