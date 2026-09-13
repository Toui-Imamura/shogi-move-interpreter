"""
interpreter/feature_vector.py

F01〜F40の特徴量を固定長ベクトルへ変換する。

特徴量の順序:
    [F01, F02, ..., F40]

学習済み重みと入力ベクトルの対応を崩さないため、
特徴量の順序はこのモジュールで一元管理する。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


FEATURE_NAMES: tuple[str, ...] = tuple(
    f"F{i:02d}" for i in range(1, 41)
)

FEATURE_COUNT = len(FEATURE_NAMES)


def feature_names() -> tuple[str, ...]:
    """
    F01〜F40の固定順序を返す。
    """
    return FEATURE_NAMES


def dict_to_vector(
    values: Mapping[str, float],
    *,
    default: float = 0.0,
    strict: bool = False,
) -> list[float]:
    """
    特徴量辞書をF01〜F40の固定長ベクトルへ変換する。

    Parameters
    ----------
    values:
        特徴量名をキーとする辞書。

    default:
        存在しない特徴量に設定する値。

    strict:
        Trueの場合、未知の特徴量名が存在すればエラーにする。

    Returns
    -------
    list[float]
        40次元の特徴量ベクトル。
    """
    unknown_names = set(values.keys()) - set(FEATURE_NAMES)

    if strict and unknown_names:
        raise KeyError(
            f"未知の特徴量名です: {sorted(unknown_names)}"
        )

    return [
        float(values.get(name, default))
        for name in FEATURE_NAMES
    ]


def vector_to_dict(
    vector: Sequence[float],
) -> dict[str, float]:
    """
    40次元ベクトルを特徴量辞書へ変換する。
    """
    if len(vector) != FEATURE_COUNT:
        raise ValueError(
            f"ベクトル長は{FEATURE_COUNT}である必要があります。"
            f"実際の長さ: {len(vector)}"
        )

    return {
        name: float(value)
        for name, value in zip(FEATURE_NAMES, vector)
    }
