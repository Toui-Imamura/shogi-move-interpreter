"""
interpreter/dataset_records.py

特徴量パイプラインの結果を学習用データセットレコードへ変換する。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from interpreter.feature_vector import FEATURE_COUNT
from interpreter.pipeline_result_adapter import (
    pipeline_result_to_dict,
    pipeline_result_to_vector,
)


@dataclass(frozen=True)
class FeatureDatasetRecord:
    """
    1局面・1指し手分の学習用レコード。

    features:
        F01〜F40の固定順序ベクトル。

    feature_dict:
        確認・デバッグ用の特徴量辞書。
    """

    game_id: str
    source_file: str | None
    ply: int
    side_to_move: int | None
    move_usi: str
    features: list[float]
    feature_dict: dict[str, float]
    nnue_value: float | None = None
    game_result: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        JSON保存用の辞書へ変換する。
        """
        return asdict(self)


def make_dataset_record(
    *,
    game_id: str,
    source_file: str | None,
    ply: int,
    side_to_move: int | None,
    move_usi: str,
    pipeline_result: Any,
    nnue_value: float | None = None,
    game_result: int | None = None,
) -> FeatureDatasetRecord:
    """
    FeaturePipelineResultからデータセットレコードを生成する。
    """

    feature_dict = pipeline_result_to_dict(
        pipeline_result,
    )

    features = pipeline_result_to_vector(
        pipeline_result,
    )

    if len(features) != FEATURE_COUNT:
        raise ValueError(
            f"特徴量ベクトルは{FEATURE_COUNT}次元である必要があります。"
            f"実際の次元数: {len(features)}"
        )

    return FeatureDatasetRecord(
        game_id=str(game_id),
        source_file=source_file,
        ply=int(ply),
        side_to_move=(
            None
            if side_to_move is None
            else int(side_to_move)
        ),
        move_usi=str(move_usi),
        features=[float(value) for value in features],
        feature_dict={
            name: float(value)
            for name, value in feature_dict.items()
        },
        nnue_value=(
            None
            if nnue_value is None
            else float(nnue_value)
        ),
        game_result=(
            None
            if game_result is None
            else int(game_result)
        ),
    )


def record_to_json_dict(
    record: FeatureDatasetRecord,
) -> dict[str, Any]:
    """
    レコードをJSONシリアライズ可能な辞書へ変換する。
    """
    return record.to_dict()
