from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cshogi

from interpreter.explainer import explain_move


CSA_MOVE_PATTERN = re.compile(r"^([+-])(\d{4}[A-Z]{2})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a CSA kifu and generate move explanations."
    )
    parser.add_argument("csa_path", type=Path)
    parser.add_argument("--start-ply", type=int, default=1)
    parser.add_argument("--end-ply", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
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
    return parser.parse_args()


def load_csa_moves(csa_path: Path) -> list[tuple[str, str]]:
    moves: list[tuple[str, str]] = []

    with csa_path.open("r", encoding="shift_jis") as f:
        for raw_line in f:
            line = raw_line.strip()

            match = CSA_MOVE_PATTERN.match(line)
            if match is None:
                continue

            color = match.group(1)
            csa_move = match.group(2)
            moves.append((color, csa_move))

    return moves


def build_result(
    *,
    ply: int,
    csa: str,
    move_usi: str,
    sfen_before: str,
    sfen_after: str,
    explanation,
) -> dict[str, object]:
    """Build a JSON-serializable research result."""

    candidates = [
        {
            "priority": candidate.priority,
            "feature_name": candidate.feature_name,
            "text": candidate.text,
            "score": candidate.score,
        }
        for candidate in explanation.candidates
    ]

    return {
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


def print_compact_result(
    *,
    ply: int,
    move_usi: str,
    explanation: str,
) -> None:
    """Print the human-readable compact format."""

    print(f"[{ply}] {move_usi}")
    print(f"    {explanation}")


def print_verbose_result(
    *,
    result: dict[str, object],
) -> None:
    """Print detailed information for research/debugging."""

    print("=" * 80)
    print(f"PLY: {result['ply']}")
    print(f"CSA: {result['csa']}")
    print(f"USI: {result['usi']}")
    print(f"SFEN before: {result['sfen_before']}")
    print(f"SFEN after:  {result['sfen_after']}")

    print("Candidates:")
    candidates = result["candidates"]

    if candidates:
        for candidate in candidates:
            print(
                "  "
                f"{candidate['feature_name']}: "
                f"{candidate['text']} "
                f"(priority={candidate['priority']}, "
                f"score={candidate['score']:.4f})"
            )
    else:
        print("  none")

    print("Feature deltas:")
    deltas = result["normalized_deltas"]

    changed = [
        (name, value)
        for name, value in deltas.items()
        if abs(value) > 0.05
    ]

    if not changed:
        print("  none")
    else:
        for name, value in sorted(changed):
            print(f"  {name}: {value:+.4f}")

    print("Explanation:")
    print(f"  {result['explanation']}")


def main() -> None:
    args = parse_args()

    if not args.csa_path.exists():
        raise FileNotFoundError(
            f"CSA file not found: {args.csa_path}"
        )

    if args.start_ply < 1:
        raise ValueError("--start-ply must be >= 1")

    if args.end_ply is not None and args.end_ply < args.start_ply:
        raise ValueError("--end-ply must be >= --start-ply")

    csa_moves = load_csa_moves(args.csa_path)

    if not csa_moves:
        raise ValueError("No CSA moves were found.")

    end_ply = args.end_ply or len(csa_moves)
    selected_moves = csa_moves[args.start_ply - 1 : end_ply]

    board = cshogi.Board()
    results: list[dict[str, object]] = []

    for index, (color, csa_move) in enumerate(
        selected_moves,
        start=args.start_ply,
    ):
        before_sfen = board.sfen()

        # cshogi.move_from_csa() expects the move without
        # the CSA color prefix (+/-).
        move = board.move_from_csa(csa_move)

        if move == 0:
            raise ValueError(
                f"Invalid CSA move at ply {index}: "
                f"{color}{csa_move}"
            )

        if not board.is_legal(move):
            raise ValueError(
                f"Illegal move at ply {index}: "
                f"{color}{csa_move}"
            )

        move_usi = cshogi.move_to_usi(move)

        explanation = explain_move(
            board,
            move_usi,
        )

        board.push(move)
        after_sfen = board.sfen()

        result = build_result(
            ply=index,
            csa=f"{color}{csa_move}",
            move_usi=move_usi,
            sfen_before=before_sfen,
            sfen_after=after_sfen,
            explanation=explanation,
        )

        results.append(result)

        if args.verbose:
            print_verbose_result(result=result)
        else:
            print_compact_result(
                ply=index,
                move_usi=move_usi,
                explanation=explanation.explanation,
            )

        if args.show_candidates and not args.verbose:
            for candidate in result["candidates"]:
                print(
                    f"    - {candidate['feature_name']}: "
                    f"{candidate['text']}"
                )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)

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
        print(f"Saved: {args.output}")

    print()
    print(
        f"Processed {len(results)} moves "
        f"(ply {args.start_ply}-{args.start_ply + len(results) - 1})."
    )


if __name__ == "__main__":
    main()
