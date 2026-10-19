"""指し手の前後で攻撃・守備候補の変化を比較する。"""

from dataclasses import dataclass

from interpreter.square_strategy import (
    SquareCandidate,
    analyze_square_strategy,
)


@dataclass(frozen=True)
class SquareStrategyChange:
    """1マス・1候補種別における指し手前後の変化。"""

    square: int
    usi: str
    kind: str
    before_score: float
    after_score: float
    delta: float
    status: str
    before_reasons: tuple[str, ...]
    after_reasons: tuple[str, ...]


def _candidate_map(candidates):
    return {
        (candidate.square, candidate.kind): candidate
        for candidate in candidates
    }


def compare_square_strategy(
    before,
    after,
    color: int,
    *,
    top_n: int = 10,
) -> tuple[SquareStrategyChange, ...]:
    """
    同じ陣営について、指し手前後の候補評価を比較する。

    status:
        appeared  : 指し手後に候補として出現
        disappeared: 指し手後に候補から消失
        changed   : 候補は両方に存在し、評価値が変化
    """
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    before_groups = analyze_square_strategy(before, color, top_n=81)
    after_groups = analyze_square_strategy(after, color, top_n=81)

    before_map = {}
    after_map = {}

    for group in before_groups:
        before_map.update(_candidate_map(group))
    for group in after_groups:
        after_map.update(_candidate_map(group))

    changes = []

    for key in before_map.keys() | after_map.keys():
        old = before_map.get(key)
        new = after_map.get(key)

        old_score = old.score if old is not None else 0.0
        new_score = new.score if new is not None else 0.0
        delta = new_score - old_score

        if old is None:
            status = "appeared"
        elif new is None:
            status = "disappeared"
        else:
            status = "changed"

        if abs(delta) < 1e-9 and status == "changed":
            continue

        candidate = new if new is not None else old
        assert candidate is not None

        changes.append(
            SquareStrategyChange(
                square=candidate.square,
                usi=candidate.usi,
                kind=candidate.kind,
                before_score=old_score,
                after_score=new_score,
                delta=delta,
                status=status,
                before_reasons=old.reasons if old else (),
                after_reasons=new.reasons if new else (),
            )
        )

    # 絶対変化量の大きい順に並べる。
    changes.sort(key=lambda item: (-abs(item.delta), item.usi, item.kind))
    return tuple(changes[:top_n])


def print_square_strategy_changes(
    before,
    after,
    color: int,
    *,
    top_n: int = 10,
) -> None:
    """指し手前後で変化した候補を表示する。"""
    changes = compare_square_strategy(
        before,
        after,
        color,
        top_n=top_n,
    )

    print("\n【指し手前後の攻撃・守備候補の変化】")
    print("※ 評価値は暫定ヒューリスティックで、勝率や最善手確率ではありません。")

    if not changes:
        print("候補評価の変化はありません。")
        return

    kind_names = {
        "attack_target": "攻撃対象",
        "attack_base": "攻めの拠点",
        "defense_piece": "守備対象の味方駒",
        "defense_point": "競合地点の守備候補",
    }
    status_names = {
        "appeared": "出現",
        "disappeared": "消失",
        "changed": "評価変化",
    }

    for index, item in enumerate(changes, start=1):
        print(
            f"{index:2}. {item.usi} "
            f"[{kind_names.get(item.kind, item.kind)}] "
            f"{status_names[item.status]} "
            f"{item.before_score:.2f} -> {item.after_score:.2f} "
            f"(差分 {item.delta:+.2f})"
        )

        if item.before_reasons:
            print("    指し手前：" + "、".join(item.before_reasons))
        if item.after_reasons:
            print("    指し手後：" + "、".join(item.after_reasons))
