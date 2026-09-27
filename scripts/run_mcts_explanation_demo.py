import cshogi
import torch

from interpreter.feature_pipeline import compute_feature_pipeline
from interpreter.mcts_explanation import explain_with_mcts
from mcts.search import SimpleMCTS
from mcts.variation import convert_variations, to_mcts_feature_inputs
from training.nnue.network import NNUE
from training.nnue.features import num_feature_ids


def create_model() -> NNUE:
    """
    現時点ではMCTSパイプライン接続確認用に
    ランダム初期化されたNNUEを生成する。

    学習済みNNUEが完成したら、ここをチェックポイント
    読み込みへ変更する。
    """
    torch.manual_seed(0)

    return NNUE(
        num_features=num_feature_ids(),
        accumulator_size=256,
        hidden_size=32,
    )


def main() -> None:
    board = cshogi.Board()

    move_usi = "7g7f"
    move = board.move_from_usi(move_usi)

    if not board.is_legal(move):
        raise ValueError(f"illegal move: {move_usi}")

    print("=== Move ===")
    print(move_usi)

    model = create_model()

    print()
    print("=== MCTS ===")

    # まずは小規模探索で動作確認する。
    mcts = SimpleMCTS(
        model=model,
        simulations=30,
        max_depth=5,
        exploration_constant=1.4,
        device="cpu",
    )

    # 指し手を実行した後の局面から未来を探索する。
    after = board.copy()
    after.push(move)

    search_result = mcts.search(after)

    variations = mcts.top_variations(
        search_result,
        num_variations=3,
        max_depth=5,
    )

    print(f"variations: {len(variations)}")

    for index, variation in enumerate(variations, start=1):
        print(
            f"{index}: "
            f"visits={variation.visits}, "
            f"value={variation.value:.4f}, "
            f"moves={len(variation.moves)}"
        )

        moves_usi = [
            cshogi.move_to_usi(move)
            for move in variation.moves
        ]

        print("   ", " ".join(moves_usi))

    # MCTS variationを特徴量計算用へ変換する。
    converted = convert_variations(
        after,
        variations,
    )

    mcts_inputs = to_mcts_feature_inputs(
        converted,
    )

    # 元局面→実際の指し手→MCTS未来局面まで統合する。
    pipeline_result = compute_feature_pipeline(
        before=board,
        move=move,
        mcts_variations=mcts_inputs,
    )

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

    # MCTSを利用した説明文を生成する。
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
