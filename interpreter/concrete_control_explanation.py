from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import cshogi

from features.activity import all_attacks


_PIECE_NAMES = {
    cshogi.PAWN: "歩",
    cshogi.LANCE: "香",
    cshogi.KNIGHT: "桂",
    cshogi.SILVER: "銀",
    cshogi.BISHOP: "角",
    cshogi.ROOK: "飛",
    cshogi.GOLD: "金",
}


def square_to_usi(square: int) -> str:
    file_number = (square // 9) + 1
    rank = chr(ord("a") + (square % 9))
    return f"{file_number}{rank}"


def piece_name(
    board: cshogi.Board,
    square: int,
) -> str:
    piece = board.piece(square)

    if piece == 0:
        return "駒"

    piece_type = cshogi.piece_to_piece_type(piece)
    return _PIECE_NAMES.get(piece_type, "駒")


@dataclass(frozen=True)
class ConcreteControlEvidence:
    """
    2局面間で変化した、1駒の具体的な効き。
    """

    color: int
    piece_square: str
    piece_name: str
    added_squares: tuple[str, ...]
    removed_squares: tuple[str, ...]


@dataclass(frozen=True)
class AggregatedControlEvidence:
    """
    複数MCTS variationに共通して現れた具体的な効きの変化。
    """

    color: int
    piece_square: str
    piece_name: str
    square: str
    variation_count: int
    total_variations: int


def extract_concrete_control_evidence(
    before: cshogi.Board,
    after: cshogi.Board,
) -> tuple[ConcreteControlEvidence, ...]:
    """前後局面で変化した駒の効きを具体的なマスとして取得する。"""

    results: list[ConcreteControlEvidence] = []

    before_attacks = {
        cshogi.BLACK: all_attacks(before, cshogi.BLACK),
        cshogi.WHITE: all_attacks(before, cshogi.WHITE),
    }

    after_attacks = {
        cshogi.BLACK: all_attacks(after, cshogi.BLACK),
        cshogi.WHITE: all_attacks(after, cshogi.WHITE),
    }

    for color in (cshogi.BLACK, cshogi.WHITE):
        squares = (
            set(before_attacks[color].keys())
            | set(after_attacks[color].keys())
        )

        for square in sorted(squares):
            old_squares = before_attacks[color].get(
                square,
                set(),
            )
            new_squares = after_attacks[color].get(
                square,
                set(),
            )

            added = new_squares - old_squares
            removed = old_squares - new_squares

            if not added and not removed:
                continue

            current_piece = after.piece(square)

            if current_piece == 0:
                current_piece = before.piece(square)

            if current_piece == 0:
                continue

            results.append(
                ConcreteControlEvidence(
                    color=color,
                    piece_square=square_to_usi(square),
                    piece_name=piece_name(after, square),
                    added_squares=tuple(
                        sorted(
                            square_to_usi(item)
                            for item in added
                        )
                    ),
                    removed_squares=tuple(
                        sorted(
                            square_to_usi(item)
                            for item in removed
                        )
                    ),
                )
            )

    return tuple(results)


def aggregate_control_evidence(
    variation_evidence: list[
        tuple[ConcreteControlEvidence, ...]
    ],
    *,
    min_variation_count: int = 2,
    max_items: int = 2,
) -> tuple[AggregatedControlEvidence, ...]:
    """
    複数MCTS variationに共通して現れる効きの変化を集約する。

    各variationは1回だけカウントする。
    そのため、同じvariation内で同じ駒・同じ升への変化が
    複数回出てもvariation_countは増加しない。

    訪問回数はここでは使わない。
    variation_countを「共通性」の指標として使用する。
    """

    if min_variation_count <= 0:
        raise ValueError(
            "min_variation_count must be positive"
        )

    if max_items <= 0:
        raise ValueError(
            "max_items must be positive"
        )

    total_variations = len(variation_evidence)

    if total_variations == 0:
        return ()

    evidence_map: dict[
        tuple[int, str, str, str],
        dict[str, object],
    ] = {}

    for evidence in variation_evidence:
        seen: set[
            tuple[int, str, str, str]
        ] = set()

        for item in evidence:
            for square in item.added_squares:
                key = (
                    item.color,
                    item.piece_square,
                    item.piece_name,
                    square,
                )

                if key in seen:
                    continue

                seen.add(key)

                if key not in evidence_map:
                    evidence_map[key] = {
                        "variation_count": 0,
                    }

                evidence_map[key]["variation_count"] = (
                    int(
                        evidence_map[key][
                            "variation_count"
                        ]
                    )
                    + 1
                )

    results: list[AggregatedControlEvidence] = []

    for (
        color,
        piece_square,
        piece_name_value,
        square,
    ), data in evidence_map.items():
        variation_count = int(
            data["variation_count"]
        )

        if variation_count < min_variation_count:
            continue

        results.append(
            AggregatedControlEvidence(
                color=color,
                piece_square=piece_square,
                piece_name=piece_name_value,
                square=square,
                variation_count=variation_count,
                total_variations=total_variations,
            )
        )

    results.sort(
        key=lambda item: (
            item.variation_count,
            item.piece_square,
            item.square,
        ),
        reverse=True,
    )

    return tuple(
        results[:max_items]
    )


def build_control_sentence(
    evidence: tuple[ConcreteControlEvidence, ...],
    *,
    max_items: int = 2,
) -> str:
    """1つの局面変化を、人間が読める短い説明へ変換する。"""

    candidates = [
        item
        for item in evidence
        if item.added_squares
    ]

    if not candidates:
        return ""

    candidates.sort(
        key=lambda item: (
            -len(item.added_squares),
            item.piece_square,
        )
    )

    sentences: list[str] = []

    for item in candidates[:max_items]:
        squares = "・".join(
            item.added_squares
        )

        sentences.append(
            f"{item.piece_square}の"
            f"{item.piece_name}の効きが"
            f"{squares}へ広がります"
        )

    return (
        "また、"
        + "、".join(sentences)
        + "。"
    )


def build_aggregated_control_sentence(
    evidence: tuple[AggregatedControlEvidence, ...],
) -> str:
    """
    複数MCTS variationに共通する効きの変化を説明する。
    """

    if not evidence:
        return ""

    sentences: list[str] = []

    for item in evidence:
        sentences.append(
            f"{item.total_variations}つの展開のうち"
            f"{item.variation_count}つで"
            f"{item.piece_square}の"
            f"{item.piece_name}の効きが"
            f"{item.square}へ広がります"
        )

    if len(sentences) == 1:
        return (
            "また、MCTSの後続展開では、"
            + sentences[0]
            + "。"
        )

    return (
        "また、MCTSの後続展開では、"
        + "、".join(sentences)
        + "。"
    )
