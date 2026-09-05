"""
training/feature_dataset.py

CSAから生成した局面データに対して、
局面そのものから計算可能な特徴量を抽出する。

入力:
    data/processed/positions.jsonl

出力:
    data/processed/feature_positions.jsonl

注意:
    すべてのF01～F33が単一局面から計算できるわけではない。

    本ファイルでは「局面特徴」として直接計算可能な特徴のみ扱う。
    指し手前後の変化やVariation、MCTSを必要とする特徴は、
    別のデータ生成処理で扱う。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cshogi

from features.basic import (
    f01_material,
    f02_hand_material,
    f04_major_balance,
    f05_minor_composition,
)

from features.activity import (
    f06_control,
    f07_mobility,
    f08_important_control,
    f10_activity,
    f16_advancement,
    f17_defense_pieces,
    f18_defense_control,
)

from features.attack import (
    f11_attack_pressure,
    f12_attackers,
    f21_attack_concentration,
    f22_threat_formation,
)

from features.defense import f23_defense

from features.king_safety import (
    f26_king_safety,
)

from features.defensive_placement import (
    f27_defensive_placement,
)

from features.formation import (
    f28_formation,
)

from features.castle import (
    f29_castle_progress,
)

from features.weakness import (
    f30_weakness,
)

from features.pawn_structure import (
    f31_pawn_connectivity,
)

from features.force_distribution import (
    f33_force_distribution,
)


BLACK = cshogi.BLACK
WHITE = cshogi.WHITE


def _difference(value: Any) -> float:
    """differenceフィールドをfloatへ変換する。"""

    return float(value.difference)


def _score(value: Any) -> float:
    """scoreフィールドをfloatへ変換する。"""

    return float(value.score)


def _black_white_difference(value: Any) -> float:
    """
    black - whiteを計算する。

    black / whiteを持つ特徴量用。
    """

    return float(value.black) - float(value.white)


def extract_f01_f05(board: cshogi.Board) -> dict[str, float]:
    """
    F01～F05を抽出する。

    F03は指し手を必要とするため除外する。
    """

    result: dict[str, float] = {}

    result["F01"] = _difference(
        f01_material(board)
    )

    result["F02"] = _difference(
        f02_hand_material(board)
    )

    result["F04"] = _difference(
        f04_major_balance(board)
    )

    f05 = f05_minor_composition(board)

    # F05はdifferenceが駒種ごとの辞書。
    # 現在の仕様では、構成差の絶対値を合計して
    # 1つのスカラー値として保存する。
    result["F05"] = float(
        sum(
            abs(float(value))
            for value in f05.difference.values()
        )
    )

    return result


def extract_f06_f18(board: cshogi.Board) -> dict[str, float]:
    """
    F06～F18のうち、局面から直接計算可能な特徴量を抽出する。
    """

    result: dict[str, float] = {}

    result["F06"] = _difference(
        f06_control(board)
    )

    result["F07"] = _difference(
        f07_mobility(board)
    )

    result["F08"] = _difference(
        f08_important_control(board)
    )

    result["F10"] = _difference(
        f10_activity(board)
    )

    result["F11"] = _difference(
        f11_attack_pressure(board)
    )

    result["F12"] = _difference(
        f12_attackers(board)
    )

    result["F16"] = _difference(
        f16_advancement(board)
    )

    result["F17"] = _difference(
        f17_defense_pieces(board)
    )

    result["F18"] = _difference(
        f18_defense_control(board)
    )

    return result


def extract_f21_f23(board: cshogi.Board) -> dict[str, float]:
    """
    F21～F23を局面特徴として抽出する。
    """

    result: dict[str, float] = {}

    # F21
    black_f21 = f21_attack_concentration(
        board,
        BLACK,
    )

    white_f21 = f21_attack_concentration(
        board,
        WHITE,
    )

    result["F21"] = (
        float(black_f21)
        - float(white_f21)
    )

    # F22
    black_f22 = f22_threat_formation(
        board,
        BLACK,
    )

    white_f22 = f22_threat_formation(
        board,
        WHITE,
    )

    result["F22"] = (
        float(black_f22)
        - float(white_f22)
    )

    # F23
    black_f23 = f23_defense(
        board,
        BLACK,
    )

    white_f23 = f23_defense(
        board,
        WHITE,
    )

    result["F23"] = (
        _score(black_f23)
        - _score(white_f23)
    )

    return result


def extract_f26_f33(
    board: cshogi.Board,
) -> dict[str, float]:
    """
    F26～F33のうち、
    単一局面から直接計算可能な特徴量を抽出する。

    F32は指し手前後の局面が必要なため、
    この関数では扱わない。
    """

    result: dict[str, float] = {}

    # F26: 玉の安全度
    result["F26"] = _difference(
        f26_king_safety(board)
    )

    # F27: 守備駒配置
    black_f27 = f27_defensive_placement(
        board,
        BLACK,
    )

    white_f27 = f27_defensive_placement(
        board,
        WHITE,
    )

    result["F27"] = (
        _score(black_f27)
        - _score(white_f27)
    )

    # F28: 陣形
    black_f28 = f28_formation(
        board,
        BLACK,
    )

    white_f28 = f28_formation(
        board,
        WHITE,
    )

    result["F28"] = (
        _score(black_f28)
        - _score(white_f28)
    )

    # F29: キャッスル進展度
    black_f29 = f29_castle_progress(
        board,
        BLACK,
    )

    white_f29 = f29_castle_progress(
        board,
        WHITE,
    )

    result["F29"] = (
        _score(black_f29)
        - _score(white_f29)
    )

    # F30: 弱点
    black_f30 = f30_weakness(
        board,
        BLACK,
    )

    white_f30 = f30_weakness(
        board,
        WHITE,
    )

    result["F30"] = (
        _score(black_f30)
        - _score(white_f30)
    )

    # F31: 歩の連結性
    black_f31 = f31_pawn_connectivity(
        board,
        BLACK,
    )

    white_f31 = f31_pawn_connectivity(
        board,
        WHITE,
    )

    result["F31"] = (
        _score(black_f31)
        - _score(white_f31)
    )

    # F33: 戦力分布
    black_f33 = f33_force_distribution(
        board,
        BLACK,
    )

    white_f33 = f33_force_distribution(
        board,
        WHITE,
    )

    result["F33"] = (
        _score(black_f33)
        - _score(white_f33)
    )

    return result


def extract_position_features(
    board: cshogi.Board,
) -> dict[str, float]:
    """
    1局面から直接計算可能な特徴量を抽出する。

    戻り値:
        {
            "F01": ...,
            "F02": ...,
            ...
        }

    注意:
        F03, F09, F13, F14, F15, F19, F20,
        F24, F25 ,F32は「局面単体」ではなく、
        指し手前後やVariationを必要とするため、
        この関数では扱わない。
    """

    result: dict[str, float] = {}

    result.update(
        extract_f01_f05(board)
    )

    result.update(
        extract_f06_f18(board)
    )

    result.update(
        extract_f21_f23(board)
    )

    result.update(
        extract_f26_f33(board)
    )

    return dict(sorted(result.items()))


def convert_position_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """
    positions.jsonlの1レコードに特徴量を追加する。
    """

    if "sfen" not in record:
        raise ValueError(
            "record does not contain 'sfen'"
        )

    board = cshogi.Board(
        sfen=record["sfen"]
    )

    features = extract_position_features(
        board
    )

    return {
        **record,
        "features": features,
    }


def convert_jsonl_to_feature_jsonl(
    input_path: str | Path,
    output_path: str | Path,
) -> int:
    """
    positions.jsonlから
    feature_positions.jsonlを生成する。
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with (
        input_path.open(
            "r",
            encoding="utf-8",
        ) as input_file,
        output_path.open(
            "w",
            encoding="utf-8",
        ) as output_file,
    ):
        for line in input_file:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            feature_record = (
                convert_position_record(record)
            )

            output_file.write(
                json.dumps(
                    feature_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            count += 1

    return count