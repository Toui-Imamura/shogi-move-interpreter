"""
attack.py

F11～F15:
    F11 相手玉への攻撃圧力
    F12 攻撃参加駒数
    F13 駒得形成過程
    F14 攻撃拠点形成
    F15 攻撃継続性

F11/F12:
    現在局面Sにおける攻撃状態を評価する。

F13～F15:
    S0 -> S1 -> ... -> Sv の変化列を利用して評価する。

注意:
    本モジュールの攻撃指標は「指し手の意図」を直接表すものではない。
    あくまで局面上に現れる攻撃状態・変化を数値化する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import cshogi

from .activity import (
    BLACK,
    WHITE,
    all_attacks,
    control_on_squares,
    controlled_squares,
    king_area,
    opponent,
    piece_color,
    square_to_file_rank,
)
from .common import (
    board_material,
    king_square,
    tanh_normalize,
)
from .transition import Variation


# ---------------------------------------------------------------------------
# 共通攻撃領域
# ---------------------------------------------------------------------------

def enemy_king_area(
    board: cshogi.Board,
    color: int,
    radius: int = 1,
) -> Set[int]:
    """
    colorから見た相手玉周辺のマス集合。
    """
    return king_area(
        board,
        opponent(color),
        radius=radius,
    )


def invasion_squares(
    board: cshogi.Board,
    color: int,
) -> Set[int]:
    """
    color側から見た侵入候補マス。

    Black:
        rank 0～2を相手陣とする。

    White:
        rank 6～8を相手陣とする。

    cshogiの座標系に依存するため、activity.pyで確認済みの
    file/rank変換を利用する。
    """
    result: Set[int] = set()

    for square in range(81):
        _, rank = square_to_file_rank(square)

        if color == BLACK and rank <= 2:
            result.add(square)

        elif color == WHITE and rank >= 6:
            result.add(square)

    return result


def attack_base_candidates(
    board: cshogi.Board,
    color: int,
) -> Set[int]:
    """
    攻撃拠点候補を求める。

    現段階では、
        1. 相手玉周辺
        2. 相手陣
    を候補とする。

    F14では、この候補のうち複数の駒から利きがある地点を
    実際の攻撃拠点として扱う。
    """
    return (
        enemy_king_area(board, color)
        | invasion_squares(board, color)
    )


# ---------------------------------------------------------------------------
# F11 相手玉への攻撃圧力
# ---------------------------------------------------------------------------

def king_attackers(
    board: cshogi.Board,
    color: int,
) -> Set[int]:
    """
    相手玉周辺を攻撃している駒の位置集合。
    """
    targets = enemy_king_area(board, color)

    result: Set[int] = set()

    for square, attacks in all_attacks(board, color).items():
        if attacks.intersection(targets):
            result.add(square)

    return result


def king_control_count(
    board: cshogi.Board,
    color: int,
) -> int:
    """
    相手玉周辺に対する利き数。

    同一マスへの複数の利きは複数として数える。
    """
    targets = enemy_king_area(board, color)

    return control_on_squares(
        board,
        color,
        targets,
    )


def invasion_control_count(
    board: cshogi.Board,
    color: int,
) -> int:
    """
    相手陣への利き数。
    """
    targets = invasion_squares(board, color)

    return control_on_squares(
        board,
        color,
        targets,
    )


@dataclass(frozen=True)
class F11AttackPressure:
    black: float
    white: float
    difference: float

    black_king_control: int
    white_king_control: int

    black_attackers: int
    white_attackers: int

    black_invasion: int
    white_invasion: int


def _raw_attack_pressure(
    board: cshogi.Board,
    color: int,
) -> float:
    """
    攻撃圧力の生値。

    Pressure =
        a1 * KingControl
      + a2 * Attackers
      + a3 * Invasion

    係数は現段階では暫定値。
    """
    king_control = king_control_count(board, color)
    attackers = len(king_attackers(board, color))
    invasion = invasion_control_count(board, color)

    a1 = 1.0
    a2 = 2.0
    a3 = 0.5

    return (
        a1 * king_control
        + a2 * attackers
        + a3 * invasion
    )


def f11_attack_pressure(
    board: cshogi.Board,
    normalization_c: float = 10.0,
) -> F11AttackPressure:
    """
    F11 相手玉への攻撃圧力。

    王手そのものだけではなく、

        ・相手玉周辺への利き
        ・攻撃参加駒数
        ・相手陣への侵入力

    を組み合わせる。
    """
    black_king_control = king_control_count(board, BLACK)
    white_king_control = king_control_count(board, WHITE)

    black_attackers = len(
        king_attackers(board, BLACK)
    )
    white_attackers = len(
        king_attackers(board, WHITE)
    )

    black_invasion = invasion_control_count(
        board,
        BLACK,
    )
    white_invasion = invasion_control_count(
        board,
        WHITE,
    )

    black_raw = _raw_attack_pressure(
        board,
        BLACK,
    )
    white_raw = _raw_attack_pressure(
        board,
        WHITE,
    )

    black = tanh_normalize(
        black_raw,
        normalization_c,
    )
    white = tanh_normalize(
        white_raw,
        normalization_c,
    )

    return F11AttackPressure(
        black=black,
        white=white,
        difference=black - white,
        black_king_control=black_king_control,
        white_king_control=white_king_control,
        black_attackers=black_attackers,
        white_attackers=white_attackers,
        black_invasion=black_invasion,
        white_invasion=white_invasion,
    )


# ---------------------------------------------------------------------------
# F12 攻撃参加駒数
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class F12Attackers:
    black: int
    white: int
    difference: int


def attacking_piece_set(
    board: cshogi.Board,
    color: int,
) -> Set[int]:
    """
    攻撃参加駒の位置集合。

    以下のいずれかを満たす駒を攻撃参加駒とする。

        1. 相手玉周辺を利いている
        2. 相手陣を利いている

    同じ駒は1回だけ数える。
    """
    king_targets = enemy_king_area(board, color)
    invasion_targets = invasion_squares(board, color)

    result: Set[int] = set()

    for square, attacks in all_attacks(board, color).items():
        if attacks.intersection(king_targets):
            result.add(square)
            continue

        if attacks.intersection(invasion_targets):
            result.add(square)

    return result


def f12_attackers(
    board: cshogi.Board,
) -> F12Attackers:
    """
    F12 攻撃参加駒数。
    """
    black = len(
        attacking_piece_set(board, BLACK)
    )
    white = len(
        attacking_piece_set(board, WHITE)
    )

    return F12Attackers(
        black=black,
        white=white,
        difference=black - white,
    )


# ---------------------------------------------------------------------------
# F13 駒得形成過程
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class F13MaterialGainProcess:
    black: float
    white: float
    difference: float

    black_positive_gain: float
    white_positive_gain: float

    steps: int


def _material_balance(
    board: cshogi.Board,
) -> float:
    """
    Black - Whiteの盤上駒価値差。

    common.board_material() は陣営を指定して
    盤上の駒価値を返すため、Black / Whiteを
    個別に計算して差を取る。
    """
    black_material = board_material(
        board,
        BLACK,
    )

    white_material = board_material(
        board,
        WHITE,
    )

    return float(
        black_material - white_material
    )


def f13_material_gain_process(
    variation: Variation,
) -> F13MaterialGainProcess:
    """
    F13 駒得形成過程。

    仕様:
        GainProcess = sum_t max(0, ΔF01_t)

    ここでは各時点の駒得差について、
    「その時点で増加した側」の増加量を累積する。

    Black:
        Blackのmaterial balanceが増加した分だけ累積。

    White:
        White側に有利になる方向の増加を累積。
    """
    positions = variation.positions

    if len(positions) <= 1:
        return F13MaterialGainProcess(
            black=0.0,
            white=0.0,
            difference=0.0,
            black_positive_gain=0.0,
            white_positive_gain=0.0,
            steps=0,
        )

    black_gain = 0.0
    white_gain = 0.0

    previous = _material_balance(
        positions[0]
    )

    for board in positions[1:]:
        current = _material_balance(board)
        delta = current - previous

        if delta > 0:
            black_gain += delta

        elif delta < 0:
            white_gain += -delta

        previous = current

    return F13MaterialGainProcess(
        black=black_gain,
        white=white_gain,
        difference=black_gain - white_gain,
        black_positive_gain=black_gain,
        white_positive_gain=white_gain,
        steps=len(positions) - 1,
    )


# ---------------------------------------------------------------------------
# F14 攻撃拠点形成
# ---------------------------------------------------------------------------

def attack_base_control_counts(
    board: cshogi.Board,
    color: int,
) -> Dict[int, int]:
    """
    攻撃拠点候補ごとの利き数を返す。
    """
    candidates = attack_base_candidates(
        board,
        color,
    )

    result: Dict[int, int] = {}

    for square in candidates:
        count = 0

        for attacks in all_attacks(
            board,
            color,
        ).values():
            if square in attacks:
                count += 1

        result[square] = count

    return result


def attack_base_set(
    board: cshogi.Board,
    color: int,
    minimum_attackers: int = 2,
) -> Set[int]:
    """
    攻撃拠点として成立しているマス。

    現段階では、
        「同陣営の2駒以上が利いている候補地点」
    と定義する。
    """
    counts = attack_base_control_counts(
        board,
        color,
    )

    return {
        square
        for square, count in counts.items()
        if count >= minimum_attackers
    }


@dataclass(frozen=True)
class F14AttackBase:
    black: float
    white: float
    difference: float

    black_bases: int
    white_bases: int


def f14_attack_base(
    before: cshogi.Board,
    after: cshogi.Board,
) -> F14AttackBase:
    """
    F14 攻撃拠点形成。

    BaseGain =
        攻撃拠点数(Sv) - 攻撃拠点数(S0)

    正:
        攻撃拠点が形成された

    負:
        攻撃拠点が失われた
    """
    black_before = len(
        attack_base_set(before, BLACK)
    )
    black_after = len(
        attack_base_set(after, BLACK)
    )

    white_before = len(
        attack_base_set(before, WHITE)
    )
    white_after = len(
        attack_base_set(after, WHITE)
    )

    black_change = black_after - black_before
    white_change = white_after - white_before

    return F14AttackBase(
        black=float(black_change),
        white=float(white_change),
        difference=float(
            black_change - white_change
        ),
        black_bases=black_change,
        white_bases=white_change,
    )


# ---------------------------------------------------------------------------
# F15 攻撃継続性
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class F15AttackContinuity:
    black: float
    white: float
    difference: float

    black_active_steps: int
    white_active_steps: int

    steps: int


def f15_attack_continuity(
    variation: Variation,
    threshold: float = 0.0,
) -> F15AttackContinuity:
    """
    F15 攻撃継続性。

    Variation中に攻撃圧力が一定以上存在した割合を求める。

        Continuity =
            攻撃圧力がthreshold以上だった局面数
            / 評価対象局面数

    MCTSの1 variationについて評価する。
    """
    positions = variation.positions

    if not positions:
        return F15AttackContinuity(
            black=0.0,
            white=0.0,
            difference=0.0,
            black_active_steps=0,
            white_active_steps=0,
            steps=0,
        )

    black_active = 0
    white_active = 0

    for board in positions:
        pressure = f11_attack_pressure(board)

        if pressure.black > threshold:
            black_active += 1

        if pressure.white > threshold:
            white_active += 1

    steps = len(positions)

    black = black_active / steps
    white = white_active / steps

    return F15AttackContinuity(
        black=black,
        white=white,
        difference=black - white,
        black_active_steps=black_active,
        white_active_steps=white_active,
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Combined extraction
# ---------------------------------------------------------------------------

def extract_attack_features(
    board: cshogi.Board,
) -> Dict[str, object]:
    """
    現在局面からF11/F12を抽出する。
    """
    return {
        "F11": f11_attack_pressure(board),
        "F12": f12_attackers(board),
    }
