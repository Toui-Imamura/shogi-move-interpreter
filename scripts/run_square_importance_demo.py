from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cshogi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.square_importance import (
    build_square_effects,
    square_to_usi,
)


def effect_map(board: cshogi.Board) -> dict[int, object]:
    return {
        effect.square: effect
        for effect in build_square_effects(board)
    }


def describe_square(effect, before_board, after_board) -> str:
    """計算できた利きの変化を、断定しすぎない文章にする。"""
    black_delta = effect.black_controls
    white_delta = effect.white_controls

    labels = []
    if effect.contested:
        labels.append("双方の利きが重なる地点")
    if black_delta > 0 and white_delta == 0:
        labels.append("先手側の利きがある地点")
    elif white_delta > 0 and black_delta == 0:
        labels.append("後手側の利きがある地点")

    occupied = after_board.piece(effect.square) != 0
    if occupied:
        labels.append("駒が配置されている地点")
    else:
        labels.append("空きマス")

    return "、".join(labels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将棋の重要地点と指し手前後の変化を表示する"
    )
    parser.add_argument(
        "--sfen",
        default=None,
        help="解析する局面のSFEN。省略時は初期局面。",
    )
    parser.add_argument(
        "--move",
        default="7g7f",
        help="解析する指し手のUSI表記。初期値: 7g7f",
    )
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top は1以上にしてください")

    before = cshogi.Board(sfen=args.sfen) if args.sfen else cshogi.Board()
    legal_moves = {
        cshogi.move_to_usi(move): move
        for move in before.legal_moves
    }

    if args.move not in legal_moves:
        parser.error(
            f"指し手 {args.move} は指定局面で合法手ではありません"
        )

    before_effects = effect_map(before)
    after = before.copy()
    after.push_usi(args.move)
    after_effects = effect_map(after)

    print("=" * 68)
    print("将棋の重要地点・指し手解釈デモ")
    print("=" * 68)
    print(f"指し手: {args.move}")
    print(f"変更前SFEN: {before.sfen()}")
    print(f"変更後SFEN: {after.sfen()}")
    print()
    print("【1】指し手の前後で重要度が変化した地点")
    print("重要度は既存実装のヒューリスティック値です。勝率ではありません。")

    changes = []
    for square in range(81):
        old = before_effects[square]
        new = after_effects[square]
        delta = new.importance - old.importance

        if delta != 0:
            changes.append((square, old, new, delta))

    changes.sort(key=lambda item: abs(item[3]), reverse=True)

    if not changes:
        print("重要度の変化はありませんでした。")
    else:
        print("マス   前   後   増減   先手利き(前→後)   後手利き(前→後)")
        for square, old, new, delta in changes[:args.top]:
            print(
                f"{square_to_usi(square):4} "
                f"{old.importance:4} "
                f"{new.importance:4} "
                f"{delta:+5} "
                f"{old.black_controls:5}→{new.black_controls:<5} "
                f"{old.white_controls:5}→{new.white_controls:<5}"
            )

    print()
    print("【2】変更後の重要地点 上位")
    ranked = sorted(
        after_effects.values(),
        key=lambda e: (
            e.importance,
            e.black_controls + e.white_controls,
        ),
        reverse=True,
    )

    for rank, effect in enumerate(ranked[:args.top], start=1):
        print(
            f"{rank:2}. {effect.usi} "
            f"重要度={effect.importance} "
            f"先手利き={effect.black_controls} "
            f"後手利き={effect.white_controls} "
            f"{'双方の利きあり' if effect.contested else ''}"
        )

    print()
    print("【3】地点の解釈")
    if not changes:
        print("今回の指し手では、重要度変化に基づく説明候補を抽出できませんでした。")
    else:
        for square, old, new, delta in changes[:args.top]:
            usi = square_to_usi(square)
            if delta > 0:
                meaning = "重要度が上昇"
            else:
                meaning = "重要度が低下"

            black_change = new.black_controls - old.black_controls
            white_change = new.white_controls - old.white_controls

            reasons = []
            if black_change > 0:
                reasons.append(f"先手側の利きが{black_change}増加")
            elif black_change < 0:
                reasons.append(f"先手側の利きが{abs(black_change)}減少")

            if white_change > 0:
                reasons.append(f"後手側の利きが{white_change}増加")
            elif white_change < 0:
                reasons.append(f"後手側の利きが{abs(white_change)}減少")

            if new.contested:
                reasons.append("変更後は双方の利きが重なる")

            reason_text = "、".join(reasons) if reasons else (
                "利きの数以外の重要度要因が変化"
            )
            print(f"- {usi}: {meaning}（{old.importance}→{new.importance}）。{reason_text}。")

    print()
    print("注意: 利きの数は戦略的な重要性や勝率そのものではありません。")
    print("また、利きがあることは、そのマスへの着手が合法・有利であることを保証しません。")


if __name__ == "__main__":
    main()
