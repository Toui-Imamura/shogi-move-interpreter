"""
interpreter/mcts_explanation.py

MCTSの探索結果を利用して、
対象の指し手を人間向けの文章として説明する。

基本構造:

    S0 --対象手--> S1
                    |
                    +-- MCTS --> S2 ... Sv
                    |
                    +-- F37/F38/F39
                    |
                    +-- 説明文

説明では、

1. 対象手の具体的な移動
2. 対象手による即時の特徴量変化
3. MCTSで複数variationに現れた特徴量変化
4. F37/F38/F39

を組み合わせる。

MCTSの訪問回数は、
「その意図の確率」ではなく、
複数variationを集約するための重みとして扱う。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cshogi

from interpreter.explanation_rules import (
    generate_explanation_candidates,
    select_top_explanation_candidates,
    compose_explanation,
)

from interpreter.feature_pipeline import (
    UnifiedFeaturePipelineResult,
)


# ============================================================
# Piece information
# ============================================================

_PIECE_NAMES = {
    cshogi.PAWN: "歩",
    cshogi.LANCE: "香",
    cshogi.KNIGHT: "桂",
    cshogi.SILVER: "銀",
    cshogi.BISHOP: "角",
    cshogi.ROOK: "飛",
    cshogi.GOLD: "金",
}


def _piece_name_from_square(
    board: cshogi.Board,
    square: int,
) -> str:
    """
    指定升にある駒の名称を返す。

    cshogi 1.0.4では board.piece(square) は
    整数の駒コードを返すため、cshogi.piece_to_piece_type()
    で駒種へ変換する。
    """

    piece = board.piece(square)

    if piece == 0:
        return "駒"

    piece_type = cshogi.piece_to_piece_type(piece)

    return _PIECE_NAMES.get(
        piece_type,
        "駒",
    )


def _square_to_usi(square: int) -> str:
    """
    cshogiの升番号をUSIの升表記へ変換する。

    square:
        0〜80

    例:
        7g -> USI上の文字列表現
    """

    file_number = 9 - (square % 9)
    rank_number = square // 9

    rank_letters = "abcdefghi"

    return (
        f"{file_number}"
        f"{rank_letters[rank_number]}"
    )


# ============================================================
# Move description
# ============================================================


@dataclass(frozen=True)
class MoveDescription:
    """
    対象手の具体的な説明情報。
    """

    move_usi: str
    piece_name: str
    from_square: str | None
    to_square: str
    is_drop: bool


def describe_move(
    board: cshogi.Board,
    move: int,
) -> MoveDescription:
    """
    cshogiのmoveから、
    駒種・移動元・移動先を取得する。

    通常手:
        7g7f
        -> 7gから7fへ歩を移動

    駒打ち:
        P*7f
        -> 持ち駒の歩を7fへ打つ
    """

    move_usi = cshogi.move_to_usi(move)

    # 駒打ち
    if "*" in move_usi:
        piece_letter, to_square = move_usi.split("*")

        drop_piece_names = {
            "P": "歩",
            "L": "香",
            "N": "桂",
            "S": "銀",
            "G": "金",
            "B": "角",
            "R": "飛",
        }

        return MoveDescription(
            move_usi=move_usi,
            piece_name=drop_piece_names.get(
                piece_letter,
                "駒",
            ),
            from_square=None,
            to_square=to_square,
            is_drop=True,
        )

    from_square = move_usi[:2]
    to_square = move_usi[2:4]

    from_file = int(from_square[0])
    from_rank = from_square[1]

    from_square_index = (
        (ord(from_rank) - ord("a")) * 9
        + (9 - from_file)
    )

    piece_name = _piece_name_from_square(
        board,
        from_square_index,
    )

    return MoveDescription(
        move_usi=move_usi,
        piece_name=piece_name,
        from_square=from_square,
        to_square=to_square,
        is_drop=False,
    )


# ============================================================
# MCTS common changes
# ============================================================


@dataclass(frozen=True)
class MCTSChange:
    """
    MCTSによって得られた特徴量変化。
    """

    feature_name: str
    change: float


def extract_mcts_changes(
    pipeline_result: UnifiedFeaturePipelineResult,
    *,
    threshold: float = 0.05,
    max_features: int = 5,
) -> list[MCTSChange]:
    """
    MCTSの訪問重み付き変化から、
    説明に使用する特徴量を抽出する。

    F38のweighted_changesを利用する。

    訪問回数そのものを確率とは解釈しない。
    """

    if pipeline_result is None:
        raise ValueError(
            "pipeline_result must not be None"
        )

    mcts = pipeline_result.mcts

    if mcts is None:
        return []

    weighted_changes = mcts.f38.weighted_changes

    candidates: list[MCTSChange] = []

    for feature_name, change in weighted_changes.items():
        change = float(change)

        if abs(change) < threshold:
            continue

        candidates.append(
            MCTSChange(
                feature_name=feature_name,
                change=change,
            )
        )

    candidates.sort(
        key=lambda item: abs(item.change),
        reverse=True,
    )

    return candidates[:max_features]


# ============================================================
# Explanation
# ============================================================


@dataclass(frozen=True)
class MCTSExplanation:
    """
    MCTSを利用した指し手解釈結果。
    """

    move: MoveDescription

    immediate_candidates: Sequence

    mcts_changes: Sequence[MCTSChange]

    explanation: str

    f37_frequency: float | None

    f39_concentration: float | None


def _build_move_sentence(
    move: MoveDescription,
) -> str:
    """
    対象手そのものを説明する。
    """

    if move.is_drop:
        return (
            f"{move.piece_name}を"
            f"{move.to_square}へ打つ手です。"
        )

    return (
        f"{move.piece_name}を"
        f"{move.from_square}から"
        f"{move.to_square}へ"
        f"動かす手です。"
    )


def _feature_name_to_text(
    feature_name: str,
    change: float,
) -> str | None:
    """
    MCTS特徴量を説明文へ変換する。

    ここではMCTS固有のF37〜F39ではなく、
    F38が示した「元の特徴量」の意味を説明する。
    """

    positive = change > 0

    texts = {
        "F01": (
            "駒得を拡大する"
            if positive
            else "駒得を縮小する"
        ),
        "F02": (
            "持ち駒の価値を高める"
            if positive
            else "持ち駒の価値を低下させる"
        ),
        "F06": (
            "盤面上の利きや支配範囲を変化させる"
        ),
        "F07": (
            "駒の可動性を高める"
            if positive
            else "駒の可動性を制限する"
        ),
        "F08": (
            "重要な地点の制御を強める"
            if positive
            else "重要な地点の制御を弱める"
        ),
        "F09": (
            "働きの弱い駒を活用しやすくする"
            if positive
            else "駒の働きを制限する"
        ),
        "F10": (
            "駒の活動性を高める"
            if positive
            else "駒の活動性を低下させる"
        ),
        "F11": (
            "相手玉への攻撃圧力を高める"
            if positive
            else "相手玉への攻撃圧力を弱める"
        ),
        "F12": (
            "攻撃に参加する駒を増やす"
            if positive
            else "攻撃に参加する駒を減らす"
        ),
        "F14": (
            "攻撃の拠点を形成する"
            if positive
            else "攻撃の拠点を失う"
        ),
        "F16": (
            "駒を前進させて攻撃や守備に参加させる"
            if positive
            else "駒の前進を抑える"
        ),
        "F17": (
            "玉周辺の守備駒を増やす"
            if positive
            else "玉周辺の守備駒を減らす"
        ),
        "F18": (
            "玉周辺の制御を強化する"
            if positive
            else "玉周辺の制御を弱める"
        ),
        "F19": (
            "相手陣への圧力を高める"
            if positive
            else "相手陣への圧力を弱める"
        ),
        "F20": (
            "攻撃駒の数を増やす"
            if positive
            else "攻撃駒の数を減らす"
        ),
        "F21": (
            "攻撃を特定の地点に集中させる"
            if positive
            else "攻撃の集中を弱める"
        ),
        "F22": (
            "相手に対する脅威を形成する"
            if positive
            else "脅威を弱める"
        ),
        "F23": (
            "守備力を高める"
            if positive
            else "守備力を低下させる"
        ),
        "F24": (
            "玉の安全性を高める"
            if positive
            else "玉の安全性を低下させる"
        ),
        "F25": (
            "相手の攻撃への対応を強める"
            if positive
            else "相手の攻撃への対応を弱める"
        ),
        "F26": (
            "玉の安全性を改善する"
            if positive
            else "玉の安全性を低下させる"
        ),
        "F27": (
            "守備駒の配置を改善する"
            if positive
            else "守備駒の配置を崩す"
        ),
        "F28": (
            "陣形を整える"
            if positive
            else "陣形を崩す"
        ),
        "F29": (
            "囲いを進展させる"
            if positive
            else "囲いを後退させる"
        ),
        "F30": (
            "自陣の弱点を補う"
            if positive
            else "自陣に弱点を生じさせる"
        ),
        "F31": (
            "歩の連結性を改善する"
            if positive
            else "歩の連結性を弱める"
        ),
        "F32": (
            "攻守のバランスを変化させる"
        ),
        "F33": (
            "盤面上の戦力配置を改善する"
            if positive
            else "盤面上の戦力配置を変化させる"
        ),
    }

    return texts.get(feature_name)


def _build_mcts_sentence(
    changes: Sequence[MCTSChange],
) -> str:
    """
    MCTS変化から説明文を作る。
    """

    texts: list[str] = []

    for change in changes:
        text = _feature_name_to_text(
            change.feature_name,
            change.change,
        )

        if text is None:
            continue

        texts.append(text)

    if not texts:
        return ""

    if len(texts) == 1:
        return (
            "MCTSでは、この手の後に"
            f"{texts[0]}変化が確認されます。"
        )

    joined = "、".join(texts[:3])

    return (
        "MCTSでは、この手の後に"
        f"{joined}といった変化が複数の展開で確認されます。"
    )


def explain_with_mcts(
    board: cshogi.Board,
    move: int,
    pipeline_result: UnifiedFeaturePipelineResult,
    *,
    max_candidates: int = 3,
    mcts_threshold: float = 0.05,
) -> MCTSExplanation:
    """
    対象手とMCTS結果を組み合わせて説明する。
    """

    if board is None:
        raise ValueError(
            "board must not be None"
        )

    if pipeline_result is None:
        raise ValueError(
            "pipeline_result must not be None"
        )

    move_description = describe_move(
        board,
        move,
    )

    immediate_deltas = (
        pipeline_result.normalized_deltas
    )

    immediate_candidates = (
        generate_explanation_candidates(
            immediate_deltas
        )
    )

    selected_candidates = (
        select_top_explanation_candidates(
            immediate_candidates,
            max_candidates=max_candidates,
        )
    )

    base_explanation = compose_explanation(
        move_description.move_usi,
        selected_candidates,
        include_note=False,
    )

    mcts_changes = extract_mcts_changes(
        pipeline_result,
        threshold=mcts_threshold,
    )

    move_sentence = _build_move_sentence(
        move_description
    )

    mcts_sentence = _build_mcts_sentence(
        mcts_changes
    )

    parts = [
        move_sentence,
        base_explanation,
    ]

    if mcts_sentence:
        parts.append(mcts_sentence)

    explanation = " ".join(
        part.strip()
        for part in parts
        if part and part.strip()
    )

    f37_frequency = None
    f39_concentration = None

    if pipeline_result.mcts is not None:
        feature_frequencies = (
            pipeline_result.mcts.f37.feature_frequencies
        )

        f37_frequency = max(
            feature_frequencies.values(),
            default=0.0,
        )

        f39_concentration = float(
            pipeline_result.mcts.f39
        )

    return MCTSExplanation(
        move=move_description,
        immediate_candidates=selected_candidates,
        mcts_changes=mcts_changes,
        explanation=explanation,
        f37_frequency=f37_frequency,
        f39_concentration=f39_concentration,
    )
