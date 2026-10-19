import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cshogi

from features.square_importance import (
    changed_square_effects,
    top_important_squares,
)
from interpreter.mcts_square_importance import (
    analyze_pipeline_square_importance,
)
from interpreter.square_strategy_change import (
    compare_square_strategy,
)

from interpreter.explainer import explain_move
from interpreter.mcts_explanation import (
    describe_mcts_variations,
    extract_feature_transition_evidence,
    explain_with_mcts,
)
from interpreter.feature_pipeline import compute_feature_pipeline
from mcts.search import SimpleMCTS
from mcts.variation import (
    convert_variations,
    to_mcts_feature_inputs,
)
from mcts.yaneuraou_evaluator import YaneuraOuMCTSEvaluator


CSA_MOVE_PATTERN = re.compile(r"^([+-])(\d{4}[A-Z]{2})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a CSA kifu and generate move explanations."
    )

    parser.add_argument(
        "csa_path",
        type=Path,
    )

    parser.add_argument(
        "--start-ply",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--end-ply",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed SFEN and feature-change information.",
    )

    parser.add_argument(
        "--show-candidates",
        action="store_true",
        help="Show explanation candidates in addition to the main explanation.",
    )

    parser.add_argument(
        "--mcts",
        action="store_true",
        help="Run MCTS-based move interpretation.",
    )

    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=30,
        help="Number of MCTS simulations per move.",
    )

    parser.add_argument(
        "--mcts-depth",
        type=int,
        default=5,
        help="Maximum MCTS variation depth.",
    )

    parser.add_argument(
        "--mcts-variations",
        type=int,
        default=3,
        help="Number of top MCTS variations.",
    )

    parser.add_argument(
        "--yaneuraou-engine",
        type=Path,
        default=Path(
            "/workspace/YaneuraOu/source/YaneuraOu-by-gcc"
        ),
        help="Path to the YaneuraOu executable.",
    )

    parser.add_argument(
        "--yaneuraou-eval-dir",
        type=Path,
        default=Path(
            "/workspace/YaneuraOu/source/eval"
        ),
        help="Directory containing YaneuraOu NNUE evaluation files.",
    )

    parser.add_argument(
        "--yaneuraou-depth",
        type=int,
        default=8,
        help="YaneuraOu search depth used for each MCTS evaluation.",
    )

    parser.add_argument(
        "--yaneuraou-threads",
        type=int,
        default=1,
        help="Number of YaneuraOu search threads.",
    )

    parser.add_argument(
        "--yaneuraou-hash",
        type=int,
        default=256,
        help="YaneuraOu hash size in MB.",
    )

    parser.add_argument(
        "--yaneuraou-timeout",
        type=int,
        default=60,
        help="Timeout in seconds for a YaneuraOu evaluation.",
    )

    return parser.parse_args()


def create_mcts_searcher(
    *,
    evaluator: YaneuraOuMCTSEvaluator,
    simulations: int,
    max_depth: int,
) -> SimpleMCTS:
    """
    YaneuraOuを評価器として利用するMCTS検索器を生成する。
    """

    if evaluator is None:
        raise ValueError(
            "evaluator must not be None"
        )

    if simulations <= 0:
        raise ValueError(
            "--mcts-simulations must be positive"
        )

    if max_depth <= 0:
        raise ValueError(
            "--mcts-depth must be positive"
        )

    return SimpleMCTS(
        evaluator=evaluator,
        simulations=simulations,
        max_depth=max_depth,
        exploration_constant=1.4,
    )

def load_csa_moves(
    csa_path: Path,
) -> list[tuple[str, str]]:
    moves: list[tuple[str, str]] = []

    with csa_path.open(
        "r",
        encoding="shift_jis",
    ) as f:
        for raw_line in f:
            line = raw_line.strip()

            match = CSA_MOVE_PATTERN.match(line)

            if match is None:
                continue

            color = match.group(1)
            csa_move = match.group(2)

            moves.append(
                (
                    color,
                    csa_move,
                )
            )

    return moves


def build_mcts_result(
    *,
    board: cshogi.Board,
    move: int,
    simulations: int,
    max_depth: int,
    num_variations: int,
    evaluator: YaneuraOuMCTSEvaluator,
):
    """
    実際の指し手後の局面からMCTSを実行し、
    F37〜F39とMCTS説明に必要な情報を生成する。

    S0 --実際の指し手--> S1 --MCTS--> Sv
    """

    if num_variations <= 0:
        raise ValueError(
            "num_variations must be positive"
        )

    # S0から実際の指し手を適用してS1を作る。
    after = board.copy()
    after.push(move)

    searcher = create_mcts_searcher(
        evaluator=evaluator,
        simulations=simulations,
        max_depth=max_depth,
    )

    search_root = searcher.search(
        after
    )

    variations = searcher.top_variations(
        search_root,
        num_variations=num_variations,
        max_depth=max_depth,
    )

    converted = convert_variations(
        after,
        variations,
    )

    mcts_inputs = to_mcts_feature_inputs(
        converted,
    )

    pipeline_result = compute_feature_pipeline(
        before=board,
        move=move,
        mcts_variations=mcts_inputs,
    )

    return (
        pipeline_result,
        variations,
    )


def build_result(
    *,
    ply: int,
    csa: str,
    move_usi: str,
    sfen_before: str,
    sfen_after: str,
    explanation,
    mcts_data=None,
) -> dict[str, object]:
    """
    JSON-serializableな研究結果を作成する。
    """

    candidates = [
        {
            "priority": candidate.priority,
            "feature_name": candidate.feature_name,
            "text": candidate.text,
            "score": candidate.score,
        }
        for candidate in explanation.candidates
    ]

    result: dict[str, object] = {
        "ply": ply,
        "csa": csa,
        "usi": move_usi,
        "sfen_before": sfen_before,
        "sfen_after": sfen_after,
        "explanation": explanation.explanation,
        "candidates": candidates,
        "normalized_deltas": {
            key: float(value)
            for key, value in explanation.result.normalized_deltas.items()
        },
    }

    if mcts_data is not None:
        result["mcts"] = mcts_data

    return result


def serialize_mcts_data(
    *,
    pipeline_result,
    variations,
) -> dict[str, object]:
    """
    MCTS結果をJSON保存用の形式へ変換する。
    """

    descriptions = describe_mcts_variations(
        pipeline_result
    )

    serialized_variations = []

    for description, variation in zip(
        descriptions,
        variations,
    ):
        serialized_variations.append(
            {
                "variation_index": description.variation_index,
                "visits": description.visits,
                "value": float(variation.value),
                "moves": [
                    {
                        "usi": move.move_usi,
                        "piece_name": move.piece_name,
                        "from_square": move.from_square,
                        "to_square": move.to_square,
                        "is_drop": move.is_drop,
                    }
                    for move in description.moves
                ],
            }
        )

    mcts = pipeline_result.mcts

    if mcts is None:
        return {
            "variations": serialized_variations,
            "feature_frequencies": {},
            "weighted_changes": {},
            "concentration": None,
        }

    evidence_by_feature: dict[str, list[dict[str, object]]] = {}

    for feature_name in mcts.f38.weighted_changes:
        evidence = extract_feature_transition_evidence(
            pipeline_result,
            feature_name,
            threshold=0.05,
        )

        evidence_by_feature[feature_name] = [
            {
                "variation_index": item.variation_index,
                "move_index": item.move_index,
                "usi": item.move.move_usi,
                "piece_name": item.move.piece_name,
                "from_square": item.move.from_square,
                "to_square": item.move.to_square,
                "change": float(item.change),
            }
            for item in evidence
        ]

    return {
        "variations": serialized_variations,
        "feature_frequencies": {
            key: float(value)
            for key, value in mcts.f37.feature_frequencies.items()
        },
        "weighted_changes": {
            key: float(value)
            for key, value in mcts.f38.weighted_changes.items()
        },
        "concentration": float(mcts.f39),
        "evidence": evidence_by_feature,
    }



def serialize_square_analysis(
    *,
    before,
    after,
    mover_color: int,
    pipeline_result=None,
) -> dict[str, object]:
    """指し手前後の重要地点分析をJSON用に変換する。"""

    def serialize_effect(effect) -> dict[str, object]:
        return {
            "square": int(effect.square),
            "usi": effect.usi,
            "piece": int(effect.piece),
            "black_attackers": int(effect.black_attackers),
            "white_attackers": int(effect.white_attackers),
            "black_controls": int(effect.black_controls),
            "white_controls": int(effect.white_controls),
            "contested": bool(effect.contested),
            "importance": float(effect.importance),
        }

    before_top = top_important_squares(
        before, top_n=10, include_occupied=True
    )
    after_top = top_important_squares(
        after, top_n=10, include_occupied=True
    )

    effect_changes = changed_square_effects(before, after)
    immediate_changes = []

    for old, new in effect_changes:
        change_types = []

        if old.piece != new.piece:
            change_types.append("piece_placement")

        if (
            old.black_attackers != new.black_attackers
            or old.white_attackers != new.white_attackers
        ):
            change_types.append("control_change")

        if old.contested != new.contested:
            change_types.append("contested_change")

        if old.importance != new.importance:
            change_types.append("importance_change")

        immediate_changes.append({
            "square": int(new.square),
            "usi": new.usi,
            "change_types": change_types,
            "before": serialize_effect(old),
            "after": serialize_effect(new),
            "importance_delta": float(
                new.importance - old.importance
            ),
            "black_attackers_delta": int(
                new.black_attackers - old.black_attackers
            ),
            "white_attackers_delta": int(
                new.white_attackers - old.white_attackers
            ),
            "contested_before": bool(old.contested),
            "contested_after": bool(new.contested),
        })

    immediate_changes.sort(
        key=lambda item: (
            abs(item["importance_delta"]),
            abs(item["black_attackers_delta"])
            + abs(item["white_attackers_delta"]),
        ),
        reverse=True,
    )

    strategy_changes = compare_square_strategy(
        before,
        after,
        mover_color,
        top_n=8,
    )

    serialized_strategy = [
        {
            "square": int(item.square),
            "usi": item.usi,
            "kind": item.kind,
            "before_score": float(item.before_score),
            "after_score": float(item.after_score),
            "delta": float(item.delta),
            "status": item.status,
            "before_reasons": list(item.before_reasons),
            "after_reasons": list(item.after_reasons),
        }
        for item in strategy_changes
    ]

    mcts_square_data = None

    if (
        pipeline_result is not None
        and getattr(pipeline_result, "mcts", None) is not None
    ):
        mcts_result = analyze_pipeline_square_importance(
            pipeline_result
        )
        aggregated = sorted(
            mcts_result.aggregated,
            key=lambda item: abs(item.weighted_delta),
            reverse=True,
        )

        mcts_square_data = {
            "aggregation_note": (
                "訪問回数は変化を集約するための重みであり、"
                "指し手の意図や正しさの確率ではない"
            ),
            "aggregated": [
                {
                    "square": int(item.square),
                    "usi": item.usi,
                    "variation_count": int(item.variation_count),
                    "total_visits": int(item.total_visits),
                    "weighted_delta": float(item.weighted_delta),
                    "mean_delta": float(item.mean_delta),
                    "increase_count": int(item.increase_count),
                    "decrease_count": int(item.decrease_count),
                }
                for item in aggregated[:10]
            ],
        }

    return {
        "important_squares_before": [
            serialize_effect(item) for item in before_top
        ],
        "important_squares_after": [
            serialize_effect(item) for item in after_top
        ],
        "immediate_changes": immediate_changes[:15],
        "strategy_changes": serialized_strategy,
        "mcts": mcts_square_data,
    }


def summarize_square_analysis(
    square_analysis: dict[str, object],
) -> str:
    """
    重要地点の候補評価変化を、意図を断定せずに短く要約する。

    候補の出現・消失だけを指し手の意図とみなさず、
    評価差を観測結果として記述する。
    """

    kind_names = {
        "attack_target": "攻撃対象",
        "attack_base": "攻めの拠点",
        "defense_piece": "守備対象の味方駒",
        "defense_point": "守備候補地点",
    }

    parts = []
    strategy = square_analysis.get("strategy_changes", [])

    # 評価差の大きい候補から、変化が一定以上のものを1件だけ使う。
    # 評価値は暫定ヒューリスティックであり、勝率や意図の確率ではない。
    for item in strategy:
        delta = float(item.get("delta", 0.0))
        if abs(delta) < 1.0:
            continue

        kind = kind_names.get(item.get("kind"), item.get("kind", "候補"))
        direction = "上昇" if delta > 0 else "低下"
        before_score = float(item.get("before_score", 0.0))
        after_score = float(item.get("after_score", 0.0))

        parts.append(
            f"{item['usi']}の{kind}の候補評価が{direction}"
            f"した（{before_score:.2f}→{after_score:.2f}）"
        )
        break

    mcts_data = square_analysis.get("mcts")
    if mcts_data is not None:
        aggregated = mcts_data.get("aggregated", [])
        if aggregated:
            item = aggregated[0]
            parts.append(
                f"MCTSの集約結果では{item['usi']}の重要度変化が大きい"
            )

    if not parts:
        return ""

    return (
        "重要地点分析（候補評価の変化）: "
        + "。".join(parts)
        + "。なお、これは評価上の変化を示すもので、"
        "指し手の意図を直接確定するものではない。"
    )


def print_compact_result(
    *,
    ply: int,
    move_usi: str,
    explanation: str,
) -> None:
    """人間向けの簡潔な出力。"""

    print(
        f"[{ply}] {move_usi}"
    )

    print(
        f"    {explanation}"
    )


def print_mcts_result(
    *,
    pipeline_result,
    variations,
) -> None:
    """
    MCTSの探索結果を人間向けに表示する。
    """

    print("MCTS variations:")

    descriptions = describe_mcts_variations(
        pipeline_result
    )

    if not descriptions:
        print("  none")
        return

    for description, variation in zip(
        descriptions,
        variations,
    ):
        print(
            f"  Variation "
            f"{description.variation_index + 1}: "
            f"visits={description.visits}, "
            f"value={variation.value:+.4f}"
        )

        if not description.moves:
            print("    none")
            continue

        for move_index, move in enumerate(
            description.moves,
            start=1,
        ):
            if move.is_drop:
                move_text = (
                    f"{move.piece_name}を"
                    f"{move.to_square}へ打つ"
                )
            else:
                move_text = (
                    f"{move.piece_name}を"
                    f"{move.from_square}から"
                    f"{move.to_square}へ"
                )

            print(
                f"    {move_index}. "
                f"{move.move_usi} "
                f"({move_text})"
            )

    mcts = pipeline_result.mcts

    if mcts is None:
        return

    print("MCTS feature changes:")

    weighted = sorted(
        mcts.f38.weighted_changes.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    changed = [
        item
        for item in weighted
        if abs(item[1]) >= 0.05
    ]

    if not changed:
        print("  none")
    else:
        for feature_name, change in changed[:10]:
            print(
                f"  {feature_name}: "
                f"{change:+.4f}"
            )

    print(
        "MCTS change concentration: "
        f"{mcts.f39:.4f}"
    )

    print("Feature transition evidence:")

    for feature_name, _ in changed[:5]:
        evidence = extract_feature_transition_evidence(
            pipeline_result,
            feature_name,
            threshold=0.05,
        )

        if not evidence:
            continue

        print(
            f"  {feature_name}:"
        )

        for item in evidence[:3]:
            move = item.move

            if move.is_drop:
                move_text = (
                    f"{move.piece_name}を"
                    f"{move.to_square}へ打つ"
                )
            else:
                move_text = (
                    f"{move.piece_name}を"
                    f"{move.from_square}から"
                    f"{move.to_square}へ"
                )

            print(
                f"    variation="
                f"{item.variation_index + 1}, "
                f"move={item.move_index + 1}: "
                f"{move_text} "
                f"({item.change:+.4f})"
            )


def print_verbose_result(
    *,
    result: dict[str, object],
) -> None:
    """詳細情報を表示する。"""

    print("=" * 80)

    print(
        f"PLY: {result['ply']}"
    )

    print(
        f"CSA: {result['csa']}"
    )

    print(
        f"USI: {result['usi']}"
    )

    print(
        f"SFEN before: "
        f"{result['sfen_before']}"
    )

    print(
        f"SFEN after:  "
        f"{result['sfen_after']}"
    )

    print("Candidates:")

    candidates = result["candidates"]

    if candidates:
        for candidate in candidates:
            print(
                "  "
                f"{candidate['feature_name']}: "
                f"{candidate['text']} "
                f"(priority="
                f"{candidate['priority']}, "
                f"score="
                f"{candidate['score']:.4f})"
            )
    else:
        print("  none")

    print("Feature deltas:")

    deltas = result[
        "normalized_deltas"
    ]

    changed = [
        (name, value)
        for name, value in deltas.items()
        if abs(value) > 0.05
    ]

    if not changed:
        print("  none")
    else:
        for name, value in sorted(
            changed
        ):
            print(
                f"  {name}: "
                f"{value:+.4f}"
            )

    mcts_data = result.get(
        "mcts"
    )

    if mcts_data is not None:
        print(
            "MCTS:"
        )

        for variation in mcts_data[
            "variations"
        ]:
            print(
                "  Variation "
                f"{variation['variation_index'] + 1}: "
                f"visits="
                f"{variation['visits']}, "
                f"value="
                f"{variation['value']:+.4f}"
            )

            moves = variation["moves"]

            for move_index, move in enumerate(
                moves,
                start=1,
            ):
                if move["is_drop"]:
                    move_text = (
                        f"{move['piece_name']}を"
                        f"{move['to_square']}へ打つ"
                    )
                else:
                    move_text = (
                        f"{move['piece_name']}を"
                        f"{move['from_square']}から"
                        f"{move['to_square']}へ"
                    )

                print(
                    f"    {move_index}. "
                    f"{move['usi']} "
                    f"({move_text})"
                )

        print(
            "  F37/F38/F39:"
        )

        print(
            "    concentration="
            f"{mcts_data['concentration']}"
        )

        weighted_changes = sorted(
            mcts_data[
                "weighted_changes"
            ].items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        for feature_name, change in weighted_changes[:10]:
            if abs(change) < 0.05:
                continue

            print(
                f"    {feature_name}: "
                f"{change:+.4f}"
            )

    print("Explanation:")

    print(
        f"  {result['explanation']}"
    )


def main() -> None:
    args = parse_args()

    if not args.csa_path.exists():
        raise FileNotFoundError(
            f"CSA file not found: "
            f"{args.csa_path}"
        )

    if args.start_ply < 1:
        raise ValueError(
            "--start-ply must be >= 1"
        )

    if (
        args.end_ply is not None
        and args.end_ply < args.start_ply
    ):
        raise ValueError(
            "--end-ply must be >= --start-ply"
        )

    if args.mcts_variations <= 0:
        raise ValueError(
            "--mcts-variations must be positive"
        )

    csa_moves = load_csa_moves(
        args.csa_path
    )

    if not csa_moves:
        raise ValueError(
            "No CSA moves were found."
        )

    end_ply = (
        args.end_ply
        or len(csa_moves)
    )

    selected_moves = csa_moves[
        args.start_ply - 1 : end_ply
    ]

    board = cshogi.Board()

    yaneuraou_evaluator = None

    if args.mcts:
        if not args.yaneuraou_engine.exists():
            raise FileNotFoundError(
                "YaneuraOu engine not found: "
                f"{args.yaneuraou_engine}"
            )

        if not args.yaneuraou_eval_dir.exists():
            raise FileNotFoundError(
                "YaneuraOu eval directory not found: "
                f"{args.yaneuraou_eval_dir}"
            )

        yaneuraou_evaluator = YaneuraOuMCTSEvaluator(
            engine_path=str(args.yaneuraou_engine),
            eval_dir=str(args.yaneuraou_eval_dir),
            depth=args.yaneuraou_depth,
            threads=args.yaneuraou_threads,
            hash_mb=args.yaneuraou_hash,
            timeout=args.yaneuraou_timeout,
        )

    results: list[
        dict[str, object]
    ] = []

    for index, (
        color,
        csa_move,
    ) in enumerate(
        selected_moves,
        start=args.start_ply,
    ):
        before_sfen = board.sfen()

        move = board.move_from_csa(
            csa_move
        )

        if move == 0:
            raise ValueError(
                f"Invalid CSA move at ply "
                f"{index}: "
                f"{color}{csa_move}"
            )

        if not board.is_legal(move):
            raise ValueError(
                f"Illegal move at ply "
                f"{index}: "
                f"{color}{csa_move}"
            )

        move_usi = cshogi.move_to_usi(
            move
        )

        # --------------------------------------------------
        # 従来の即時特徴量ベース説明
        # --------------------------------------------------

        explanation = explain_move(
            board,
            move_usi,
        )

        # --------------------------------------------------
        # MCTS
        #
        # S0 --実際の手--> S1 --MCTS--> Sv
        # --------------------------------------------------

        mcts_pipeline_result = None
        mcts_variations = None
        mcts_explanation = None

        if args.mcts:
            (
                mcts_pipeline_result,
                mcts_variations,
            ) = build_mcts_result(
                board=board,
                move=move,
                simulations=args.mcts_simulations,
                max_depth=args.mcts_depth,
                num_variations=args.mcts_variations,
                evaluator=yaneuraou_evaluator,
            )

            mcts_explanation = explain_with_mcts(
                board=board,
                move=move,
                pipeline_result=mcts_pipeline_result,
            )

        # --------------------------------------------------
        # 重要地点分析
        # board本体を進める前に、コピー上で指し手後の局面を作る。
        # --------------------------------------------------
        after_board = board.copy()
        after_board.push(move)

        square_analysis = serialize_square_analysis(
            before=board,
            after=after_board,
            mover_color=board.turn,
            pipeline_result=mcts_pipeline_result,
        )

        # 実際の指し手を実行。
        board.push(move)

        after_sfen = board.sfen()

        mcts_data = None

        if (
            mcts_pipeline_result is not None
            and mcts_variations is not None
        ):
            mcts_data = serialize_mcts_data(
                pipeline_result=mcts_pipeline_result,
                variations=mcts_variations,
            )

        # MCTSを使用する場合はMCTS説明文を主説明として保存。
        if mcts_explanation is not None:
            final_explanation = mcts_explanation.explanation
        else:
            final_explanation = explanation.explanation

        result = build_result(
            ply=index,
            csa=f"{color}{csa_move}",
            move_usi=move_usi,
            sfen_before=before_sfen,
            sfen_after=after_sfen,
            explanation=explanation,
            mcts_data=mcts_data,
        )

        square_summary = summarize_square_analysis(
            square_analysis
        )
        if square_summary:
            final_explanation = (
                f"{final_explanation} {square_summary}"
            )

        result["explanation"] = final_explanation
        result["square_analysis"] = square_analysis

        results.append(
            result
        )

        if args.verbose:
            print_verbose_result(
                result=result
            )
        else:
            print_compact_result(
                ply=index,
                move_usi=move_usi,
                explanation=final_explanation,
            )

        if (
            args.mcts
            and mcts_pipeline_result is not None
            and mcts_variations is not None
            and args.verbose
        ):
            print_mcts_result(
                pipeline_result=mcts_pipeline_result,
                variations=mcts_variations,
            )

        if (
            args.show_candidates
            and not args.verbose
        ):
            for candidate in result[
                "candidates"
            ]:
                print(
                    f"    - "
                    f"{candidate['feature_name']}: "
                    f"{candidate['text']}"
                )

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open(
            "w",
            encoding="utf-8",
        ) as f:
            for result in results:
                json.dump(
                    result,
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")

        print()
        print(
            f"Saved: {args.output}"
        )

    print()

    if results:
        print(
            f"Processed "
            f"{len(results)} moves "
            f"(ply "
            f"{args.start_ply}-"
            f"{args.start_ply + len(results) - 1})."
        )

    if yaneuraou_evaluator is not None:
        yaneuraou_evaluator.close()


if __name__ == "__main__":
    main()
