from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ExplanationCandidate:
    """
    特徴量変化から生成される説明候補。

    priority:
        候補を採用する際の優先度。大きいほど優先する。

    feature_name:
        説明の根拠となった特徴量名。

    text:
        説明本文。

    score:
        特徴量変化の強さ。
    """

    priority: int
    feature_name: str
    text: str
    score: float


def _positive_delta(
    deltas: Mapping[str, float],
    feature_name: str,
    threshold: float = 0.05,
) -> float | None:
    """
    指定特徴量が十分に増加している場合、その値を返す。
    """
    value = float(deltas.get(feature_name, 0.0))

    if value <= threshold:
        return None

    return value


def _negative_delta(
    deltas: Mapping[str, float],
    feature_name: str,
    threshold: float = 0.05,
) -> float | None:
    """
    指定特徴量が十分に減少している場合、絶対値を返す。
    """
    value = float(deltas.get(feature_name, 0.0))

    if value >= -threshold:
        return None

    return abs(value)


def generate_explanation_candidates(
    deltas: Mapping[str, float],
) -> list[ExplanationCandidate]:
    """
    F01〜F40の特徴量差分から説明候補を生成する。

    注意
    ----
    ここで生成するのは「指し手の意味」に関する候補であり、
    棋士の意図を断定するものではない。
    """

    candidates: list[ExplanationCandidate] = []

    # --------------------------------------------------------
    # 駒得・駒損
    # --------------------------------------------------------

    f01 = _positive_delta(deltas, "F01")
    if f01 is not None:
        candidates.append(
            ExplanationCandidate(
                priority=100,
                feature_name="F01",
                text="駒得を拡大する",
                score=f01,
            )
        )

    f01_loss = _negative_delta(deltas, "F01")
    if f01_loss is not None:
        candidates.append(
            ExplanationCandidate(
                priority=100,
                feature_name="F01",
                text="駒損を伴う代わりに、別の攻めや守りを得る可能性がある",
                score=f01_loss,
            )
        )

    f02 = _positive_delta(deltas, "F02")
    if f02 is not None:
        candidates.append(
            ExplanationCandidate(
                priority=90,
                feature_name="F02",
                text="持ち駒の価値を高める",
                score=f02,
            )
        )

    # --------------------------------------------------------
    # 攻撃関連
    # --------------------------------------------------------

    for feature_name, priority, text in [
        (
            "F11",
            80,
            "相手玉への攻撃圧力を高める",
        ),
        (
            "F12",
            75,
            "攻撃に参加する駒を増やす",
        ),
        (
            "F14",
            78,
            "攻撃の拠点を形成する",
        ),
        (
            "F15",
            70,
            "攻撃を継続しやすい形を作る",
        ),
        (
            "F19",
            72,
            "相手陣への圧力を高める",
        ),
        (
            "F20",
            68,
            "攻撃駒の数を増やす",
        ),
        (
            "F21",
            66,
            "攻撃を特定の地点に集中させる",
        ),
        (
            "F22",
            74,
            "相手に対する脅威を形成する",
        ),
    ]:
        value = _positive_delta(deltas, feature_name)

        if value is not None:
            candidates.append(
                ExplanationCandidate(
                    priority=priority,
                    feature_name=feature_name,
                    text=text,
                    score=value,
                )
            )

    # --------------------------------------------------------
    # 守備・玉の安全性
    # --------------------------------------------------------

    for feature_name, priority, text in [
        (
            "F17",
            76,
            "玉周辺の守備駒を増やす",
        ),
        (
            "F18",
            76,
            "玉周辺の制御を強化する",
        ),
        (
            "F23",
            74,
            "守備力を高める",
        ),
        (
            "F24",
            82,
            "玉の安全性を高める",
        ),
        (
            "F26",
            82,
            "玉の安全性を改善する",
        ),
        (
            "F27",
            70,
            "守備駒の配置を改善する",
        ),
        (
            "F29",
            65,
            "囲いを進展させる",
        ),
        (
            "F30",
            68,
            "自陣の弱点を補う",
        ),
    ]:
        value = _positive_delta(deltas, feature_name)

        if value is not None:
            candidates.append(
                ExplanationCandidate(
                    priority=priority,
                    feature_name=feature_name,
                    text=text,
                    score=value,
                )
            )

    # --------------------------------------------------------
    # 駒の活用・配置
    # --------------------------------------------------------

    for feature_name, priority, text in [
        (
            "F06",
            60,
            "盤面上の利きや支配範囲を変化させる",
        ),
        (
            "F07",
            58,
            "駒の可動性を高める",
        ),
        (
            "F08",
            58,
            "重要な地点の制御を強める",
        ),
        (
            "F09",
            62,
            "働きの弱い駒を活用しやすくする",
        ),
        (
            "F10",
            60,
            "駒の活動性を高める",
        ),
        (
            "F16",
            55,
            "駒を前進させて攻撃や守備に参加させる",
        ),
        (
            "F33",
            56,
            "盤面上の戦力配置を改善する",
        ),
    ]:
        value = _positive_delta(deltas, feature_name)

        if value is not None:
            candidates.append(
                ExplanationCandidate(
                    priority=priority,
                    feature_name=feature_name,
                    text=text,
                    score=value,
                )
            )

    # --------------------------------------------------------
    # 陣形・構造
    # --------------------------------------------------------

    for feature_name, priority, text in [
        (
            "F05",
            45,
            "駒の構成を変化させる",
        ),
        (
            "F28",
            60,
            "陣形を整える",
        ),
        (
            "F31",
            48,
            "歩の連結性を改善する",
        ),
        (
            "F32",
            65,
            "攻守のバランスを変化させる",
        ),
        (
            "F40",
            40,
            "局面の流れを変える",
        ),
    ]:
        value = _positive_delta(deltas, feature_name)

        if value is not None:
            candidates.append(
                ExplanationCandidate(
                    priority=priority,
                    feature_name=feature_name,
                    text=text,
                    score=value,
                )
            )

    # --------------------------------------------------------
    # マイナス変化
    # --------------------------------------------------------

    for feature_name, priority, text in [
        (
            "F24",
            82,
            "玉の安全性が低下する一方、攻撃などの別の目的を持つ可能性がある",
        ),
        (
            "F26",
            82,
            "玉の安全性が低下する一方、攻撃などの別の目的を持つ可能性がある",
        ),
        (
            "F23",
            74,
            "守備力が低下する代わりに、攻撃や駒の活用を優先する可能性がある",
        ),
        (
            "F07",
            58,
            "一部の駒の可動性を制限する可能性がある",
        ),
    ]:
        value = _negative_delta(deltas, feature_name)

        if value is not None:
            candidates.append(
                ExplanationCandidate(
                    priority=priority,
                    feature_name=feature_name,
                    text=text,
                    score=value,
                )
            )

    return candidates


def select_top_explanation_candidates(
    candidates: list[ExplanationCandidate],
    max_candidates: int = 3,
) -> list[ExplanationCandidate]:
    """
    説明候補から上位候補を選択する。

    単純な特徴量名の重複だけでなく、
    説明上の意味が重複する特徴量もまとめる。

    例
    ----
    F12「攻撃に参加する駒を増やす」
    F20「攻撃駒の数を増やす」

    は同時に発生しても、説明文では1つの攻撃駒数の変化として扱う。

    同様に、

    F24「玉の安全性を高める」
    F26「玉の安全性を改善する」

    も1つにまとめる。
    """

    if max_candidates <= 0:
        return []

    # 同じ意味として扱う特徴量のグループ。
    # 代表となる候補は priority / score の高いものを採用する。
    semantic_groups: dict[str, set[str]] = {
        "attack_piece_count": {"F12", "F20"},
        "king_safety": {"F24", "F26"},
    }

    feature_to_group: dict[str, str] = {}

    for group_name, feature_names in semantic_groups.items():
        for feature_name in feature_names:
            feature_to_group[feature_name] = group_name

    unique: dict[str, ExplanationCandidate] = {}

    for candidate in candidates:
        group_key = feature_to_group.get(
            candidate.feature_name,
            candidate.feature_name,
        )

        previous = unique.get(group_key)

        if previous is None:
            unique[group_key] = candidate
            continue

        if (
            candidate.priority,
            candidate.score,
        ) > (
            previous.priority,
            previous.score,
        ):
            unique[group_key] = candidate

    selected = sorted(
        unique.values(),
        key=lambda candidate: (
            -candidate.priority,
            -candidate.score,
            candidate.feature_name,
        ),
    )

    return selected[:max_candidates]


def _join_explanation_texts(
    texts: list[str],
) -> str:
    """
    説明候補を自然な日本語として結合する。

    「ほか、」「や」などを機械的に接続すると、
    候補の内容によって不自然な文章になるため、
    候補数に応じて接続方法を変える。
    """

    if not texts:
        return ""

    if len(texts) == 1:
        return texts[0]

    if len(texts) == 2:
        return f"{texts[0]}、また{texts[1]}"

    # 3個以上の場合。
    # 最大3候補を基本とするため、この形が通常の経路。
    if len(texts) == 3:
        return f"{texts[0]}、{texts[1]}、さらに{texts[2]}"

    return "、".join(texts[:-1]) + f"、さらに{texts[-1]}"


def compose_explanation(
    move_usi: str,
    candidates: list[ExplanationCandidate],
    *,
    include_note: bool = False,
) -> str:
    """
    指し手と説明候補から自然言語説明を生成する。

    Parameters
    ----------
    move_usi:
        USI形式の指し手。

    candidates:
        採用する説明候補。

    include_note:
        「特徴量の変化から推定した局面上の意味である」
        という研究上の注意書きを付けるかどうか。
    """

    if not candidates:
        explanation = (
            f"{move_usi}は、今回の特徴量分析では"
            "明確な意味を特定できませんでした。"
        )

        if include_note:
            explanation += (
                "ただし、これは特徴量の変化から"
                "推定した局面上の意味です。"
            )

        return explanation

    texts = [
        candidate.text.strip()
        for candidate in candidates
        if candidate.text.strip()
    ]

    if not texts:
        explanation = (
            f"{move_usi}は、今回の特徴量分析では"
            "明確な意味を特定できませんでした。"
        )
    else:
        body = _join_explanation_texts(texts)
        explanation = f"{move_usi}は、{body}手です。"

    if include_note:
        explanation += (
            "ただし、これは特徴量の変化から"
            "推定した局面上の意味です。"
        )

    return explanation


def explain_from_deltas(
    move_usi: str,
    deltas: Mapping[str, float],
    max_candidates: int = 3,
    *,
    include_note: bool = False,
) -> str:
    """
    特徴量差分から直接説明文を生成する。
    """

    candidates = generate_explanation_candidates(deltas)

    selected = select_top_explanation_candidates(
        candidates,
        max_candidates=max_candidates,
    )

    return compose_explanation(
        move_usi,
        selected,
        include_note=include_note,
    )
