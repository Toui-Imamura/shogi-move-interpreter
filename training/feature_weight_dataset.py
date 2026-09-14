"""
training/feature_weight_dataset.py

特徴量遷移データと教師AI評価値を対応付け、
特徴量重み学習用のデータセットを作成する。

入力:
    data/processed/feature_transitions_wcsc36.jsonl
    data/processed/teacher_transitions_wcsc36_10games.jsonl

出力:
    F01～F40の固定順序特徴量ベクトル X
    教師AIの指し手評価値変化 y

注意:
    現在のfeature_transitionsにはF01～F33の一部と
    f40が保存されている。

    未保存の特徴量は、現段階では0.0で補完するが、
    missing_featuresに欠損情報を記録する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from interpreter.dataset_io import read_jsonl


FEATURE_NAMES = tuple(
    f"F{i:02d}"
    for i in range(1, 41)
)

FEATURE_COUNT = len(FEATURE_NAMES)

JOIN_KEYS = (
    "game_id",
    "ply",
)


@dataclass(frozen=True)
class TrainingRecord:
    """
    1手分の学習用レコード。
    """

    game_id: int
    ply: int
    sfen: str
    move: str
    features: tuple[float, ...]
    target: float
    available_features: tuple[str, ...]
    missing_features: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """
        JSON保存用辞書へ変換する。
        """
        return {
            "game_id": self.game_id,
            "ply": self.ply,
            "sfen": self.sfen,
            "move": self.move,
            "features": list(self.features),
            "target": self.target,
            "available_features": list(
                self.available_features
            ),
            "missing_features": list(
                self.missing_features
            ),
        }


@dataclass(frozen=True)
class DatasetSplit:
    """
    学習用・検証用データ。
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_valid: np.ndarray
    y_valid: np.ndarray
    train_records: tuple[TrainingRecord, ...]
    valid_records: tuple[TrainingRecord, ...]


def _make_join_key(
    record: dict[str, Any],
) -> tuple[int, int]:
    """
    教師値・特徴量レコードの対応付けキーを作る。
    """
    return (
        int(record["game_id"]),
        int(record["ply"]),
    )


def _validate_common_fields(
    teacher_record: dict[str, Any],
    feature_record: dict[str, Any],
) -> None:
    """
    教師値と特徴量データの共通項目を検証する。
    """

    key = _make_join_key(feature_record)

    teacher_sfen = teacher_record.get("sfen")
    feature_sfen = feature_record.get("sfen")

    if teacher_sfen != feature_sfen:
        raise ValueError(
            "教師値データと特徴量データのSFENが一致しません: "
            f"key={key}, "
            f"teacher_sfen={teacher_sfen!r}, "
            f"feature_sfen={feature_sfen!r}"
        )

    teacher_move = teacher_record.get("move")
    feature_move = feature_record.get("move")

    if teacher_move != feature_move:
        raise ValueError(
            "教師値データと特徴量データの指し手が一致しません: "
            f"key={key}, "
            f"teacher_move={teacher_move!r}, "
            f"feature_move={feature_move!r}"
        )


def _extract_target(
    teacher_record: dict[str, Any],
) -> float:
    """
    教師値を取り出す。

    teacher_delta_mover_cpを使用する。

    正の値:
        指し手を指した側から見て評価が改善。

    負の値:
        指し手を指した側から見て評価が悪化。
    """

    value = teacher_record.get(
        "teacher_delta_mover_cp"
    )

    if value is None:
        raise ValueError(
            "teacher_delta_mover_cpが存在しません: "
            f"key={_make_join_key(teacher_record)}"
        )

    return float(value)


def _extract_features(
    feature_record: dict[str, Any],
) -> tuple[
    tuple[float, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """
    F01～F40の固定順序ベクトルを作る。

    F01～F39:
        feature_deltasから取得。

    F40:
        feature_record["f40"]から取得。

    現時点で存在しない特徴量は0.0で補完する。
    """

    feature_deltas = feature_record.get(
        "feature_deltas"
    )

    if not isinstance(feature_deltas, dict):
        raise ValueError(
            "feature_deltasは辞書である必要があります: "
            f"key={_make_join_key(feature_record)}"
        )

    available: list[str] = []
    missing: list[str] = []
    values: list[float] = []

    for feature_name in FEATURE_NAMES:
        if feature_name == "F40":
            if "f40" in feature_record:
                value = feature_record["f40"]
                available.append(feature_name)
            else:
                value = 0.0
                missing.append(feature_name)
        elif feature_name in feature_deltas:
            value = feature_deltas[feature_name]
            available.append(feature_name)
        else:
            value = 0.0
            missing.append(feature_name)

        try:
            values.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{feature_name}の値をfloatへ変換できません: "
                f"key={_make_join_key(feature_record)}, "
                f"value={value!r}"
            ) from exc

    return (
        tuple(values),
        tuple(available),
        tuple(missing),
    )


def load_training_records(
    *,
    teacher_path: str | Path,
    feature_path: str | Path,
    strict: bool = True,
) -> tuple[TrainingRecord, ...]:
    """
    教師値データと特徴量データを対応付ける。

    Parameters
    ----------
    teacher_path:
        教師値JSONL。

    feature_path:
        特徴量JSONL。

    strict:
        Trueの場合、SFENや指し手の不一致、
        対応する特徴量の欠如で例外を送出する。

        Falseの場合、不一致レコードをスキップする。

    Notes
    -----
    教師値データを基準に結合する。

    教師値が10ゲーム分しかない場合でも、
    特徴量データ全体から対応するレコードだけを抽出する。
    """

    feature_records: dict[
        tuple[int, int],
        dict[str, Any],
    ] = {}

    for record in read_jsonl(feature_path):
        key = _make_join_key(record)

        if key in feature_records:
            raise ValueError(
                f"特徴量データに重複キーがあります: {key}"
            )

        feature_records[key] = record

    joined_records: list[TrainingRecord] = []

    for teacher_record in read_jsonl(teacher_path):
        key = _make_join_key(teacher_record)
        feature_record = feature_records.get(key)

        if feature_record is None:
            if strict:
                raise ValueError(
                    "対応する特徴量レコードがありません: "
                    f"key={key}"
                )
            continue

        try:
            _validate_common_fields(
                teacher_record,
                feature_record,
            )

            features, available, missing = (
                _extract_features(feature_record)
            )

            target = _extract_target(
                teacher_record
            )

            joined_records.append(
                TrainingRecord(
                    game_id=key[0],
                    ply=key[1],
                    sfen=str(
                        teacher_record["sfen"]
                    ),
                    move=str(
                        teacher_record["move"]
                    ),
                    features=features,
                    target=target,
                    available_features=available,
                    missing_features=missing,
                )
            )

        except (KeyError, TypeError, ValueError):
            if strict:
                raise
            continue

    return tuple(joined_records)


def records_to_arrays(
    records: (
        tuple[TrainingRecord, ...]
        | list[TrainingRecord]
    ),
) -> tuple[np.ndarray, np.ndarray]:
    """
    学習用レコードをXとyへ変換する。
    """

    if not records:
        raise ValueError(
            "学習用レコードが空です。"
        )

    x = np.asarray(
        [
            record.features
            for record in records
        ],
        dtype=np.float32,
    )

    y = np.asarray(
        [
            record.target
            for record in records
        ],
        dtype=np.float32,
    )

    if x.ndim != 2:
        raise ValueError(
            "Xは2次元配列である必要があります: "
            f"shape={x.shape}"
        )

    if x.shape[1] != FEATURE_COUNT:
        raise ValueError(
            "Xの特徴量数が不正です: "
            f"shape={x.shape}"
        )

    if y.ndim != 1:
        raise ValueError(
            "yは1次元配列である必要があります: "
            f"shape={y.shape}"
        )

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "Xとyの件数が一致しません: "
            f"x={x.shape[0]}, y={y.shape[0]}"
        )

    return x, y


def split_records(
    records: (
        tuple[TrainingRecord, ...]
        | list[TrainingRecord]
    ),
    *,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> DatasetSplit:
    """
    学習用・検証用に分割する。

    注意:
        現段階ではレコード単位で分割する。

        同一対局の局面が訓練・検証の両方に
        入る可能性がある。

        本格的な研究評価ではgame_id単位の分割を
        別途実装する必要がある。
    """

    records = tuple(records)

    if len(records) < 2:
        raise ValueError(
            "分割には少なくとも2件のレコードが必要です。"
        )

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "validation_ratioは0と1の間で指定してください。"
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    valid_count = max(
        1,
        int(round(
            len(records) * validation_ratio
        )),
    )

    if valid_count >= len(records):
        valid_count = len(records) - 1

    valid_indices = indices[:valid_count]
    train_indices = indices[valid_count:]

    train_records = tuple(
        records[index]
        for index in train_indices
    )

    valid_records = tuple(
        records[index]
        for index in valid_indices
    )

    x_train, y_train = records_to_arrays(
        train_records
    )

    x_valid, y_valid = records_to_arrays(
        valid_records
    )

    return DatasetSplit(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        train_records=train_records,
        valid_records=valid_records,
    )


def summarize_records(
    records: (
        tuple[TrainingRecord, ...]
        | list[TrainingRecord]
    ),
) -> dict[str, Any]:
    """
    学習用レコードの概要を返す。
    """

    records = tuple(records)

    missing_counts = {
        feature_name: 0
        for feature_name in FEATURE_NAMES
    }

    for record in records:
        for feature_name in record.missing_features:
            missing_counts[feature_name] += 1

    targets = np.asarray(
        [
            record.target
            for record in records
        ],
        dtype=np.float64,
    )

    return {
        "record_count": len(records),
        "feature_count": FEATURE_COUNT,
        "target_min": (
            None
            if len(targets) == 0
            else float(targets.min())
        ),
        "target_max": (
            None
            if len(targets) == 0
            else float(targets.max())
        ),
        "target_mean": (
            None
            if len(targets) == 0
            else float(targets.mean())
        ),
        "missing_feature_counts": missing_counts,
    }
