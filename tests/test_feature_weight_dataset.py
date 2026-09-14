"""
tests/test_feature_weight_dataset.py

training.feature_weight_datasetのテスト。
"""

from pathlib import Path

import numpy as np
import pytest

from training.feature_weight_dataset import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    TrainingRecord,
    load_training_records,
    records_to_arrays,
    split_records,
    summarize_records,
)


def _write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    import json

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            file.write("\n")


def _teacher_record(
    *,
    game_id: int,
    ply: int,
    move: str = "7g7f",
    sfen: str = "test sfen",
    target: float = 10.0,
) -> dict:
    return {
        "game_id": game_id,
        "ply": ply,
        "sfen": sfen,
        "move": move,
        "teacher_delta_mover_cp": target,
    }


def _feature_record(
    *,
    game_id: int,
    ply: int,
    move: str = "7g7f",
    sfen: str = "test sfen",
    feature_deltas: dict | None = None,
    f40: float = 2.5,
) -> dict:
    if feature_deltas is None:
        feature_deltas = {
            "F01": 1.5,
            "F06": 3.0,
        }

    return {
        "game_id": game_id,
        "ply": ply,
        "sfen": sfen,
        "move": move,
        "feature_deltas": feature_deltas,
        "f40": f40,
    }


def test_load_training_records_joins_teacher_and_features(
    tmp_path: Path,
) -> None:
    teacher_path = tmp_path / "teacher.jsonl"
    feature_path = tmp_path / "features.jsonl"

    _write_jsonl(
        teacher_path,
        [
            _teacher_record(
                game_id=0,
                ply=1,
                target=12.0,
            ),
        ],
    )

    _write_jsonl(
        feature_path,
        [
            _feature_record(
                game_id=0,
                ply=1,
            ),
        ],
    )

    records = load_training_records(
        teacher_path=teacher_path,
        feature_path=feature_path,
    )

    assert len(records) == 1
    assert records[0].game_id == 0
    assert records[0].ply == 1
    assert records[0].target == 12.0
    assert len(records[0].features) == FEATURE_COUNT

    assert records[0].features[0] == 1.5
    assert records[0].features[5] == 3.0
    assert records[0].features[39] == 2.5
    assert records[0].features[1] == 0.0

    assert "F02" in records[0].missing_features
    assert "F01" in records[0].available_features
    assert "F40" in records[0].available_features


def test_teacher_side_is_used_as_join_base(
    tmp_path: Path,
) -> None:
    teacher_path = tmp_path / "teacher.jsonl"
    feature_path = tmp_path / "features.jsonl"

    _write_jsonl(
        teacher_path,
        [
            _teacher_record(
                game_id=0,
                ply=1,
            ),
        ],
    )

    _write_jsonl(
        feature_path,
        [
            _feature_record(
                game_id=0,
                ply=1,
            ),
            _feature_record(
                game_id=0,
                ply=2,
            ),
        ],
    )

    records = load_training_records(
        teacher_path=teacher_path,
        feature_path=feature_path,
    )

    assert len(records) == 1
    assert records[0].ply == 1


def test_records_to_arrays() -> None:
    records = (
        TrainingRecord(
            game_id=0,
            ply=1,
            sfen="sfen",
            move="7g7f",
            features=tuple(
                float(index)
                for index in range(FEATURE_COUNT)
            ),
            target=10.0,
            available_features=FEATURE_NAMES,
            missing_features=(),
        ),
    )

    x, y = records_to_arrays(records)

    assert x.shape == (1, FEATURE_COUNT)
    assert y.shape == (1,)
    assert x.dtype == np.float32
    assert y.dtype == np.float32


def test_split_records() -> None:
    records = tuple(
        TrainingRecord(
            game_id=index,
            ply=1,
            sfen=f"sfen-{index}",
            move="7g7f",
            features=tuple(
                float(index)
                for _ in range(FEATURE_COUNT)
            ),
            target=float(index),
            available_features=FEATURE_NAMES,
            missing_features=(),
        )
        for index in range(10)
    )

    split = split_records(
        records,
        validation_ratio=0.2,
        seed=42,
    )

    assert split.x_train.shape == (8, FEATURE_COUNT)
    assert split.y_train.shape == (8,)
    assert split.x_valid.shape == (2, FEATURE_COUNT)
    assert split.y_valid.shape == (2,)


def test_split_records_rejects_invalid_ratio() -> None:
    records = tuple(
        TrainingRecord(
            game_id=index,
            ply=1,
            sfen=f"sfen-{index}",
            move="7g7f",
            features=tuple(
                0.0
                for _ in range(FEATURE_COUNT)
            ),
            target=0.0,
            available_features=FEATURE_NAMES,
            missing_features=(),
        )
        for index in range(2)
    )

    with pytest.raises(ValueError):
        split_records(
            records,
            validation_ratio=0.0,
        )

    with pytest.raises(ValueError):
        split_records(
            records,
            validation_ratio=1.0,
        )


def test_sfen_mismatch_is_rejected(tmp_path: Path) -> None:
    teacher_path = tmp_path / "teacher.jsonl"
    feature_path = tmp_path / "features.jsonl"

    _write_jsonl(
        teacher_path,
        [
            _teacher_record(
                game_id=0,
                ply=1,
                sfen="teacher sfen",
            ),
        ],
    )

    _write_jsonl(
        feature_path,
        [
            _feature_record(
                game_id=0,
                ply=1,
                sfen="feature sfen",
            ),
        ],
    )

    with pytest.raises(ValueError, match="SFEN"):
        load_training_records(
            teacher_path=teacher_path,
            feature_path=feature_path,
        )


def test_move_mismatch_is_rejected(tmp_path: Path) -> None:
    teacher_path = tmp_path / "teacher.jsonl"
    feature_path = tmp_path / "features.jsonl"

    _write_jsonl(
        teacher_path,
        [
            _teacher_record(
                game_id=0,
                ply=1,
                move="7g7f",
            ),
        ],
    )

    _write_jsonl(
        feature_path,
        [
            _feature_record(
                game_id=0,
                ply=1,
                move="2g2f",
            ),
        ],
    )

    with pytest.raises(ValueError, match="指し手"):
        load_training_records(
            teacher_path=teacher_path,
            feature_path=feature_path,
        )


def test_missing_feature_record_is_rejected(tmp_path: Path) -> None:
    teacher_path = tmp_path / "teacher.jsonl"
    feature_path = tmp_path / "features.jsonl"

    _write_jsonl(
        teacher_path,
        [
            _teacher_record(
                game_id=0,
                ply=1,
            ),
        ],
    )

    _write_jsonl(
        feature_path,
        [],
    )

    with pytest.raises(ValueError, match="特徴量"):
        load_training_records(
            teacher_path=teacher_path,
            feature_path=feature_path,
        )


def test_f40_is_read_from_lowercase_key(
    tmp_path: Path,
) -> None:
    teacher_path = tmp_path / "teacher.jsonl"
    feature_path = tmp_path / "features.jsonl"

    _write_jsonl(
        teacher_path,
        [
            _teacher_record(
                game_id=0,
                ply=1,
            ),
        ],
    )

    _write_jsonl(
        feature_path,
        [
            _feature_record(
                game_id=0,
                ply=1,
                f40=8.75,
            ),
        ],
    )

    records = load_training_records(
        teacher_path=teacher_path,
        feature_path=feature_path,
    )

    assert records[0].features[39] == 8.75
    assert "F40" in records[0].available_features
    assert "F40" not in records[0].missing_features


def test_summarize_records() -> None:
    records = (
        TrainingRecord(
            game_id=0,
            ply=1,
            sfen="sfen",
            move="7g7f",
            features=tuple(
                0.0
                for _ in range(FEATURE_COUNT)
            ),
            target=10.0,
            available_features=("F01",),
            missing_features=tuple(
                feature_name
                for feature_name in FEATURE_NAMES
                if feature_name != "F01"
            ),
        ),
    )

    summary = summarize_records(records)

    assert summary["record_count"] == 1
    assert summary["feature_count"] == 40
    assert summary["target_min"] == 10.0
    assert summary["target_max"] == 10.0
    assert summary["target_mean"] == 10.0
    assert summary["missing_feature_counts"]["F02"] == 1
    assert summary["missing_feature_counts"]["F01"] == 0
