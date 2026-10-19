"""攻撃候補・守備候補の表示デモ。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cshogi

from interpreter.square_strategy import print_square_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sfen", default=None)
    parser.add_argument("--color", choices=("turn", "black", "white"), default="turn")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    board = cshogi.Board() if args.sfen is None else cshogi.Board(args.sfen)

    if args.color == "turn":
        color = board.turn
    elif args.color == "black":
        color = cshogi.BLACK
    else:
        color = cshogi.WHITE

    print("=== 攻撃・守備候補地点デモ ===")
    print(f"局面：{board.sfen()}")
    print_square_candidates(board, color=color, top_n=args.top)


if __name__ == "__main__":
    main()
