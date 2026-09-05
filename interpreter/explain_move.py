"""
interpreter/explain_move.py

1手の指し手について、
F01〜F40の特徴量変化を計算・表示する。

使用例:
    python -m interpreter.explain_move --move 7g7f

Variationを指定:
    python -m interpreter.explain_move \
        --move 7g7f \
        --variation 7g7f 3c3d 2g2f
"""

from __future__ import annotations

import argparse

import cshogi

from interpreter.feature_pipeline import (
    compute_feature_pipeline,
)

from features.normalization import (
    normalize_default_feature_deltas,
)


def format_value(value: float) -> str:
    """特徴量の値を表示用に整形する。"""
    return f"{value:+.6f}"


def print_immediate_features(result) -> None:
    """F01〜F33の即時変化を表示する。"""

    print()
    print("=" * 70)
    print("F01〜F33 即時特徴量変化")
    print("=" * 70)

    for feature_name, delta in result.immediate_deltas.items():
        print(
            f"{feature_name:>4}: "
            f"{format_value(delta)}"
        )


def print_variation_features(result) -> None:
    """Variation由来の特徴量を表示する。"""

    if result.variation is None:
        return

    print()
    print("=" * 70)
    print("Variation特徴量")
    print("=" * 70)

    for feature_name, value in (
        result.variation.variation_deltas.items()
    ):
        print(
            f"{feature_name:>16}: "
            f"{format_value(value)}"
        )


def print_mcts_features(result) -> None:
    """MCTS由来の特徴量を表示する。"""

    if result.mcts is None:
        return

    print()
    print("=" * 70)
    print("MCTS特徴量")
    print("=" * 70)

    print("F37 MCTS変化頻度")

    for feature_name, value in (
        result.mcts.f37.feature_frequencies.items()
    ):
        print(
            f"  {feature_name:>4}: "
            f"{value:.6f}"
        )

    print()
    print("F38 訪問重み付き変化")

    for feature_name, value in (
        result.mcts.f38.weighted_changes.items()
    ):
        print(
            f"  {feature_name:>4}: "
            f"{format_value(value)}"
        )

    print()
    print(
        "F39 変化集中度: "
        f"{result.mcts.f39:.6f}"
    )


def print_f40(result) -> None:
    """F40を表示する。"""

    print()
    print("=" * 70)
    print("F40 局面遷移度")
    print("=" * 70)

    print(
        f"F40: {result.f40_degree:.6f}"
    )


def print_top_features(result, top_n: int = 5) -> None:
    """変化量の大きい特徴量を表示する。"""

    ranked = sorted(
        result.immediate_deltas.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    print()
    print("=" * 70)
    print(f"変化量上位 {top_n} 特徴量")
    print("=" * 70)

    for feature_name, delta in ranked[:top_n]:
        print(
            f"{feature_name:>4}: "
            f"{format_value(delta)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将棋の1手について特徴量を計算する"
    )

    parser.add_argument(
        "--move",
        required=True,
        help="指し手のUSI形式。例: 7g7f",
    )

    parser.add_argument(
        "--variation",
        nargs="*",
        default=None,
        help=(
            "探索variation。"
            "例: --variation 7g7f 3c3d 2g2f"
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="表示する上位特徴量数",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    board = cshogi.Board()

    move = board.move_from_usi(args.move)

    if not board.is_legal(move):
        raise ValueError(
            f"違法な指し手です: {args.move}"
        )

    result = compute_feature_pipeline(
        before=board,
        move=move,
        variation_moves=args.variation,
    )

    print("=" * 70)
    print("将棋指し手特徴量解析")
    print("=" * 70)
    print(f"指し手: {args.move}")

    print_immediate_features(result)

    print_normalized_features(result)

    print_variation_features(result)

    print_mcts_features(result)

    print_f40(result)

    print_top_features(
        result,
        top_n=args.top,
    )


def print_normalized_features(result) -> None:
    """正規化後のF01〜F33を表示する。"""

    normalized = normalize_default_feature_deltas(
        result.immediate_deltas
    )

    print()
    print("=" * 70)
    print("正規化後 F01〜F33")
    print("=" * 70)

    for feature_name, value in normalized.items():
        print(
            f"{feature_name:>4}: "
            f"{value:+.6f}"
        )


if __name__ == "__main__":
    main()