"""MCTS variationにおける81マスの重要度変化を集計する。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import cshogi

from features.square_importance import (
    SquareEffect,
    build_square_effects,
    square_to_usi,
)


@dataclass(frozen=True)
class SquareImportanceChange:
    """1つのマスにおける、1着手前後の重要度変化。"""

    square: int
    usi: str
    before: SquareEffect
    after: SquareEffect
    delta: float

    @property
    def direction(self) -> str:
        if self.delta > 0:
            return "increase"
        if self.delta < 0:
            return "decrease"
        return "unchanged"


@dataclass(frozen=True)
class VariationSquareImportance:
    """1つの着手における重要度変化。"""

    variation_index: int
    move_index: int
    visits: int
    changes: tuple[SquareImportanceChange, ...]


@dataclass(frozen=True)
class AggregatedSquareImportance:
    """複数variationを集約した1マスの重要度変化。"""

    square: int
    usi: str
    variation_count: int
    total_visits: int
    weighted_delta: float
    mean_delta: float
    increase_count: int
    decrease_count: int


@dataclass(frozen=True)
class MCTSSquareImportanceResult:
    """MCTSの81マス重要度変化の分析結果。"""

    variations: tuple[VariationSquareImportance, ...]
    aggregated: tuple[AggregatedSquareImportance, ...]


def compare_square_importance(
    before: cshogi.Board,
    after: cshogi.Board,
    *,
    variation_index: int = 0,
    move_index: int = 0,
    visits: int = 1,
) -> VariationSquareImportance:
    """着手前後の全81マスを比較し、変化したマスを返す。"""
    if before is None or after is None:
        raise ValueError("before and after must not be None")
    if visits < 0:
        raise ValueError("visits must not be negative")

    before_effects = build_square_effects(before)
    after_effects = build_square_effects(after)

    changes = []

    for old, new in zip(before_effects, after_effects):
        delta = new.importance - old.importance

        if (
            old.piece != new.piece
            or old.black_attackers != new.black_attackers
            or old.white_attackers != new.white_attackers
            or old.importance != new.importance
        ):
            changes.append(
                SquareImportanceChange(
                    square=old.square,
                    usi=old.usi,
                    before=old,
                    after=new,
                    delta=delta,
                )
            )

    return VariationSquareImportance(
        variation_index=variation_index,
        move_index=move_index,
        visits=visits,
        changes=tuple(changes),
    )


def aggregate_square_importance(
    variations: Sequence[VariationSquareImportance],
) -> tuple[AggregatedSquareImportance, ...]:
    """
    変化をマス単位で集約する。

    variation_count:
        そのマスで変化が確認された異なるvariation数。

    total_visits:
        そのマスで変化が確認されたvariationの訪問回数の合計。
        同一variation内で同じマスが複数回変化しても重複加算しない。

    weighted_delta:
        variationごとの平均差分に訪問回数を掛けた値の合計。

    mean_delta:
        variationごとの平均差分の単純平均。

    増加・減少の回数もvariation単位で数える。
    訪問回数は集約用の重みであり、意図や正しさの確率ではない。
    """
    # variation_index -> square -> そのvariation内の差分一覧
    per_variation: dict[
        int,
        dict[int, list[SquareImportanceChange]],
    ] = defaultdict(lambda: defaultdict(list))

    visits_by_variation: dict[int, int] = {}

    for record in variations:
        previous_visits = visits_by_variation.get(record.variation_index)

        if previous_visits is not None and previous_visits != record.visits:
            raise ValueError(
                "visits must be consistent within each variation"
            )

        visits_by_variation[record.variation_index] = record.visits

        # 同一variation内の訪問回数の整合性は検証し、0回は集計しない。
        if record.visits == 0:
            continue

        for change in record.changes:
            per_variation[record.variation_index][change.square].append(change)

    # square -> variation単位の集計値
    stats: dict[int, dict[str, float | int]] = defaultdict(
        lambda: {
            "variation_count": 0,
            "total_visits": 0,
            "weighted_delta": 0.0,
            "delta_sum": 0.0,
            "increase_count": 0,
            "decrease_count": 0,
        }
    )

    for variation_index, square_changes in per_variation.items():
        visits = visits_by_variation[variation_index]

        for square, changes in square_changes.items():
            # 1つのvariation内の複数着手での差分は平均化する。
            variation_delta = sum(
                change.delta for change in changes
            ) / len(changes)

            item = stats[square]
            item["variation_count"] = int(item["variation_count"]) + 1
            item["total_visits"] = int(item["total_visits"]) + visits
            item["weighted_delta"] = (
                float(item["weighted_delta"])
                + variation_delta * visits
            )
            item["delta_sum"] = (
                float(item["delta_sum"]) + variation_delta
            )

            if variation_delta > 0:
                item["increase_count"] = (
                    int(item["increase_count"]) + 1
                )
            elif variation_delta < 0:
                item["decrease_count"] = (
                    int(item["decrease_count"]) + 1
                )

    results = []

    for square, item in stats.items():
        count = int(item["variation_count"])

        results.append(
            AggregatedSquareImportance(
                square=square,
                usi=square_to_usi(square),
                variation_count=count,
                total_visits=int(item["total_visits"]),
                weighted_delta=float(item["weighted_delta"]),
                mean_delta=(
                    float(item["delta_sum"]) / count if count else 0.0
                ),
                increase_count=int(item["increase_count"]),
                decrease_count=int(item["decrease_count"]),
            )
        )

    results.sort(
        key=lambda item: (
            abs(item.weighted_delta),
            item.variation_count,
            item.total_visits,
        ),
        reverse=True,
    )

    return tuple(results)


def analyze_mcts_square_importance(
    variation_positions: Sequence[Sequence[cshogi.Board]],
    variation_visits: Sequence[int],
) -> MCTSSquareImportanceResult:
    """
    各variationの連続局面を比較し、マス単位で集約する。

    variation_positions[i] は1つのvariationのS0, S1, ...。
    variation_visits[i] は対応するvariationの集約重み。
    """
    if len(variation_positions) != len(variation_visits):
        raise ValueError(
            "variation_positions and variation_visits must have equal length"
        )

    records = []

    for variation_index, positions in enumerate(variation_positions):
        visits = int(variation_visits[variation_index])

        if visits < 0:
            raise ValueError("variation visits must not be negative")

        # 未訪問variationは局面比較・重要度集計から除外する。
        if visits == 0:
            continue

        for move_index in range(len(positions) - 1):
            records.append(
                compare_square_importance(
                    positions[move_index],
                    positions[move_index + 1],
                    variation_index=variation_index,
                    move_index=move_index,
                    visits=visits,
                )
            )

    records_tuple = tuple(records)

    return MCTSSquareImportanceResult(
        variations=records_tuple,
        aggregated=aggregate_square_importance(records_tuple),
    )


def analyze_pipeline_square_importance(
    pipeline_result,
) -> MCTSSquareImportanceResult:
    """統合パイプラインの結果から81マスの重要度変化を分析する。"""
    if pipeline_result is None:
        raise ValueError("pipeline_result must not be None")

    mcts_result = getattr(pipeline_result, "mcts", None)
    if mcts_result is None:
        raise ValueError("pipeline_result.mcts must not be None")

    variation_results = getattr(
        mcts_result, "variation_results", None
    )
    variation_visits = getattr(
        mcts_result, "variation_visits", None
    )

    if variation_results is None:
        raise ValueError("mcts.variation_results must not be None")
    if variation_visits is None:
        raise ValueError("mcts.variation_visits must not be None")
    if len(variation_results) != len(variation_visits):
        raise ValueError(
            "variation_results and variation_visits must have equal length"
        )

    variation_positions = []

    for index, variation_result in enumerate(variation_results):
        if variation_result is None:
            raise ValueError(
                f"variation_results[{index}] must not be None"
            )

        positions = getattr(variation_result, "positions", None)
        if positions is None:
            raise ValueError(
                f"variation_results[{index}].positions must not be None"
            )

        variation_positions.append(positions)

    return analyze_mcts_square_importance(
        variation_positions=variation_positions,
        variation_visits=variation_visits,
    )
