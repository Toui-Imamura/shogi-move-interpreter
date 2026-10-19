from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cshogi

from interpreter.feature_pipeline import compute_feature_pipeline
from interpreter.mcts_explanation import explain_with_mcts
from mcts.search import SimpleMCTS
from mcts.variation import (
    convert_variations,
    to_mcts_feature_inputs,
)
from mcts.yaneuraou_evaluator import YaneuraOuMCTSEvaluator


DEFAULT_ENGINE = Path(
    "/workspace/YaneuraOu/source/YaneuraOu-by-gcc"
)
DEFAULT_EVAL_DIR = Path(
    "/workspace/YaneuraOu/source/eval"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "YaneuraOu NNUE + MCTSによる"
            "将棋の指し手解釈デモ"
        )
    )

    parser.add_argument(
        "--move",
        default="7g7f",
        help="解釈対象の指し手（USI形式）",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=3,
        help="MCTSのシミュレーション回数",
    )
    parser.add_argument(
        "--mcts-depth",
        type=int,
        default=2,
        help="MCTSの最大探索深さ",
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=3,
        help="取得する変化手順数",
    )
    parser.add_argument(
        "--engine-depth",
        type=int,
        default=1,
        help="各局面のYaneuraOu探索深さ",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="YaneuraOuの探索スレッド数",
    )
    parser.add_argument(
        "--hash",
        type=int,
        default=128,
        help="YaneuraOuのハッシュサイズ（MB）",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=DEFAULT_ENGINE,
        help="YaneuraOu実行ファイル",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help="YaneuraOuの評価ファイルディレクトリ",
    )

    args = parser.parse_args()

    if args.simulations <= 0:
        parser.error("--simulations は正の整数にしてください")
    if args.mcts_depth <= 0:
        parser.error("--mcts-depth は正の整数にしてください")
    if args.variations <= 0:
        parser.error("--variations は正の整数にしてください")
    if args.engine_depth <= 0:
        parser.error("--engine-depth は正の整数にしてください")
    if args.threads <= 0:
        parser.error("--threads は正の整数にしてください")
    if args.hash <= 0:
        parser.error("--hash は正の整数にしてください")

    return args


def main() -> None:
    args = parse_args()

    # S0: 初期局面
    board = cshogi.Board()

    move = board.move_from_usi(args.move)

    if not board.is_legal(move):
        raise ValueError(
            f"合法手ではありません: {args.move}"
        )

    print("=== Move interpretation demo ===")
    print(f"Move: {args.move}")
    print(f"MCTS simulations: {args.simulations}")
    print(f"MCTS depth: {args.mcts_depth}")
    print(f"YaneuraOu depth: {args.engine_depth}")

    # S1: 実際の指し手を適用した局面
    after = board.copy()
    after.push(move)

    print()
    print("=== YaneuraOu NNUE + MCTS ===")

    # YaneuraOuのプロセスを再利用し、終了時に確実に閉じる。
    with YaneuraOuMCTSEvaluator(
        engine_path=str(args.engine),
        eval_dir=str(args.eval_dir),
        depth=args.engine_depth,
        threads=args.threads,
        hash_mb=args.hash,
    ) as evaluator:

        # SimpleMCTSには評価器をCallableとして渡す。
        searcher = SimpleMCTS(
            evaluator=evaluator,
            simulations=args.simulations,
            max_depth=args.mcts_depth,
            exploration_constant=1.4,
        )

        # S1から未来の変化手順を探索する。
        root = searcher.search(after)

        variations = searcher.top_variations(
            root,
            num_variations=args.variations,
            max_depth=args.mcts_depth,
        )

        print(f"Variations: {len(variations)}")

        for index, variation in enumerate(
            variations,
            start=1,
        ):
            moves_usi = [
                cshogi.move_to_usi(m)
                for m in variation.moves
            ]

            print(
                f"{index}: "
                f"visits={variation.visits}, "
                f"value_black_cp={variation.value:.1f}, "
                f"moves={' '.join(moves_usi)}"
            )

        if not variations:
            print(
                "変化手順が取得できなかったため、"
                "特徴量計算をスキップします。"
            )
            return

        # MCTSの変化手順を既存の特徴量計算形式へ変換。
        converted = convert_variations(
            after,
            variations,
        )

        mcts_inputs = to_mcts_feature_inputs(
            converted,
        )

        # S0 -> S1 -> MCTS未来局面の特徴量を統合。
        pipeline_result = compute_feature_pipeline(
            before=board,
            move=move,
            mcts_variations=mcts_inputs,
        )

    # ここではYaneuraOuのプロセスを終了済み。
    # 特徴量計算結果を使った説明生成は継続できる。
    print()
    print("=== MCTS feature changes ===")

    if pipeline_result.mcts is None:
        print("MCTS feature result is None")
    else:
        print("F37 feature frequencies:")

        for feature_name, frequency in sorted(
            pipeline_result.mcts.f37.feature_frequencies.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]:
            print(
                f"  {feature_name}: {frequency:.4f}"
            )

        print(
            "F39 concentration:",
            pipeline_result.mcts.f39,
        )

        print("F38 weighted changes:")

        for feature_name, change in sorted(
            pipeline_result.mcts.f38.weighted_changes.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:10]:
            print(
                f"  {feature_name}: {change:+.4f}"
            )

    # 既存の説明生成処理を利用する。
    explanation = explain_with_mcts(
        board=board,
        move=move,
        pipeline_result=pipeline_result,
    )

    print()
    print("=== Explanation ===")
    print(explanation.explanation)

    print()
    print("=== Move description ===")
    print(
        f"{explanation.move.piece_name}を"
        f"{explanation.move.from_square}から"
        f"{explanation.move.to_square}へ動かす手"
    )

    print()
    print("=== MCTS changes used for explanation ===")

    for change in explanation.mcts_changes:
        print(
            f"{change.feature_name}: "
            f"{change.change:+.4f}"
        )


if __name__ == "__main__":
    main()
