"""指し手前後の攻撃・守備候補変化のデモ。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cshogi

from interpreter.square_strategy_change import print_square_strategy_changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfen", default=None)
    parser.add_argument("--move", default="7g7f")
    parser.add_argument("--color", choices=("black", "white"), default="black")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    board = cshogi.Board() if args.sfen is None else cshogi.Board(args.sfen)
    before = board.copy()

    color = cshogi.BLACK if args.color == "black" else cshogi.WHITE

    # 指定した手が合法手であることを確認してから適用する。
    legal_usi = {cshogi.move_to_usi(move) for move in board.legal_moves}
    if args.move not in legal_usi:
        raise SystemExit(
            f"指定した手 {args.move} はこの局面の合法手ではありません。"
        )

    board.push_usi(args.move)
    after = board

    print("=== 指し手前後の候補変化デモ ===")
    print(f"指し手：{args.move}")
    print(f"指し手前：{before.sfen()}")
    print(f"指し手後：{after.sfen()}")

    print_square_strategy_changes(
        before,
        after,
        color,
        top_n=args.top,
    )


if __name__ == "__main__":
    main()
