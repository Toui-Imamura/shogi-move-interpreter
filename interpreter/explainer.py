from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cshogi

from interpreter.explanation_rules import (
    ExplanationCandidate,
    explain_from_deltas,
    generate_explanation_candidates,
    select_top_explanation_candidates,
)
from interpreter.feature_pipeline import (
    UnifiedFeaturePipelineResult,
    compute_feature_pipeline,
)


@dataclass(frozen=True)
class MoveExplanation:
    """
    1手の特徴量分析と説明結果。
    """

    move_usi: str
    result: UnifiedFeaturePipelineResult
    candidates: tuple[ExplanationCandidate, ...]
    explanation: str

    @property
    def deltas(self) -> Mapping[str, float]:
        """
        即時特徴量差分を返す。
        """
        return self.result.immediate_deltas


def explain_move(
    before: cshogi.Board,
    move_usi: str,
    *,
    variation_moves: list[str] | None = None,
    mcts_variations: list[Mapping[str, object]] | None = None,
    feature_weights: Mapping[str, float] | None = None,
    max_candidates: int = 3,
) -> MoveExplanation:
    """
    局面とUSI指し手から説明文を生成する。

    Parameters
    ----------
    before:
        指し手前の局面。

    move_usi:
        USI形式の指し手。
        例: "7g7f"

    variation_moves:
        将来的な読み筋。
        例: ["7g7f", "3c3d", "2g2f"]

    mcts_variations:
        MCTSによる複数変化。
        現時点では任意入力として受け取る。

    feature_weights:
        F40計算に使用する特徴量重み。

    max_candidates:
        説明に採用する候補数。
    """

    if before is None:
        raise ValueError(
            "before must not be None"
        )

    if not move_usi:
        raise ValueError(
            "move_usi must not be empty"
        )

    move = before.move_from_usi(move_usi)

    result = compute_feature_pipeline(
        before=before,
        move=move,
        variation_moves=variation_moves,
        mcts_variations=mcts_variations,
        feature_weights=feature_weights,
    )

    candidates = generate_explanation_candidates(
        result.normalized_deltas
    )

    selected = select_top_explanation_candidates(
        candidates,
        max_candidates=max_candidates,
    )

    explanation = explain_from_deltas(
        move_usi=move_usi,
        deltas=result.normalized_deltas,
        max_candidates=max_candidates,
    )

    return MoveExplanation(
        move_usi=move_usi,
        result=result,
        candidates=tuple(selected),
        explanation=explanation,
    )
