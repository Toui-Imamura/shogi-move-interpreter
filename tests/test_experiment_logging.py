"""
tests/test_experiment_logging.py

experiment_logging.pyのテスト。
"""

from datetime import datetime
import json

import pytest

from training.experiment_logging import (
    append_jsonl,
    create_experiment_id,
    create_experiment_directory,
    create_unique_experiment_directory,
    initialize_experiment,
    save_experiment_config,
    save_experiment_metadata,
    save_feature_spec_snapshot,
    write_json,
)


def test_create_experiment_id():
    now = datetime(
        2026,
        9,
        14,
        12,
        34,
        56,
    )

    experiment_id = create_experiment_id(
        prefix="linear",
        now=now,
    )

    assert experiment_id == "linear_20260914_123456"


def test_create_experiment_id_sanitizes_prefix():
    now = datetime(
        2026,
        9,
        14,
        12,
        34,
        56,
    )

    experiment_id = create_experiment_id(
        prefix="test experiment/01",
        now=now,
    )

    assert experiment_id == "test_experiment_01_20260914_123456"


def test_write_json(tmp_path):
    output_path = tmp_path / "nested" / "result.json"

    write_json(
        output_path,
        {
            "name": "test",
            "value": 1.5,
        },
    )

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert data == {
        "name": "test",
        "value": 1.5,
    }


def test_append_jsonl(tmp_path):
    output_path = tmp_path / "metrics.jsonl"

    append_jsonl(
        output_path,
        {
            "epoch": 1,
            "loss": 0.5,
        },
    )
    append_jsonl(
        output_path,
        {
            "epoch": 2,
            "loss": 0.25,
        },
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == {
        "epoch": 1,
        "loss": 0.5,
    }
    assert json.loads(lines[1]) == {
        "epoch": 2,
        "loss": 0.25,
    }


def test_create_experiment_directory(tmp_path):
    experiment_dir = create_experiment_directory(
        tmp_path,
        "exp_0001",
    )

    assert experiment_dir.exists()
    assert experiment_dir.is_dir()

    with pytest.raises(FileExistsError):
        create_experiment_directory(
            tmp_path,
            "exp_0001",
        )


def test_create_unique_experiment_directory(tmp_path):
    first_id, first_dir = create_unique_experiment_directory(
        tmp_path,
        "exp_same",
    )
    second_id, second_dir = create_unique_experiment_directory(
        tmp_path,
        "exp_same",
    )

    assert first_id == "exp_same"
    assert second_id == "exp_same_001"

    assert first_dir.exists()
    assert second_dir.exists()
    assert first_dir != second_dir


def test_save_individual_experiment_files(tmp_path):
    experiment_dir = tmp_path / "exp_0001"
    experiment_dir.mkdir()

    metadata = {
        "experiment_id": "exp_0001",
        "description": "test experiment",
    }
    config = {
        "epochs": 500,
        "learning_rate": 0.01,
    }
    feature_spec = {
        "spec_version": "F01-F40-v1",
        "features": {
            "F01": {
                "name": "駒得差",
                "normalization_scale": 20.0,
            }
        },
    }

    metadata_path = save_experiment_metadata(
        experiment_dir,
        metadata,
    )
    config_path = save_experiment_config(
        experiment_dir,
        config,
    )
    feature_spec_path = save_feature_spec_snapshot(
        experiment_dir,
        feature_spec,
    )

    assert metadata_path.exists()
    assert config_path.exists()
    assert feature_spec_path.exists()

    assert json.loads(
        metadata_path.read_text(encoding="utf-8")
    ) == metadata

    assert json.loads(
        config_path.read_text(encoding="utf-8")
    ) == config

    assert json.loads(
        feature_spec_path.read_text(encoding="utf-8")
    ) == feature_spec


def test_initialize_experiment(tmp_path):
    metadata = {
        "description": "baseline experiment",
        "change_reason": "初期性能を測定する",
    }
    config = {
        "epochs": 500,
        "target_mode": "clipped_2000",
    }
    feature_spec = {
        "spec_version": "F01-F40-v1",
        "features": {
            "F01": {
                "name": "駒得差",
                "implemented": True,
            }
        },
    }

    experiment_id, experiment_dir = initialize_experiment(
        root_dir=tmp_path,
        experiment_id="exp_0001",
        metadata=metadata,
        config=config,
        feature_spec=feature_spec,
    )

    assert experiment_id == "exp_0001"
    assert experiment_dir == tmp_path / "exp_0001"

    metadata_path = experiment_dir / "experiment_metadata.json"
    config_path = experiment_dir / "config.json"
    feature_spec_path = experiment_dir / "feature_spec_snapshot.json"

    assert metadata_path.exists()
    assert config_path.exists()
    assert feature_spec_path.exists()

    saved_metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert saved_metadata["experiment_id"] == "exp_0001"
    assert saved_metadata["description"] == "baseline experiment"
    assert "created_at" in saved_metadata

    assert json.loads(
        config_path.read_text(encoding="utf-8")
    ) == config

    assert json.loads(
        feature_spec_path.read_text(encoding="utf-8")
    ) == feature_spec


def test_initialize_experiment_does_not_overwrite(tmp_path):
    metadata = {
        "description": "test",
    }
    config = {
        "epochs": 1,
    }
    feature_spec = {
        "spec_version": "test",
    }

    first_id, first_dir = initialize_experiment(
        root_dir=tmp_path,
        experiment_id="exp_same",
        metadata=metadata,
        config=config,
        feature_spec=feature_spec,
    )

    second_id, second_dir = initialize_experiment(
        root_dir=tmp_path,
        experiment_id="exp_same",
        metadata=metadata,
        config=config,
        feature_spec=feature_spec,
    )

    assert first_id == "exp_same"
    assert second_id == "exp_same_001"
    assert first_dir != second_dir


def test_save_metrics_history(tmp_path):
    from training.experiment_logging import save_metrics_history

    output_path = save_metrics_history(
        tmp_path,
        epochs=[1, 10, 20],
        train_loss=[100.0, 50.0, 25.0],
        valid_loss=[120.0, 70.0, 30.0],
        train_mae=[8.0, 6.0, 4.0],
        valid_mae=[9.0, 7.0, 5.0],
        train_rmse=[10.0, 7.0, 5.0],
        valid_rmse=[11.0, 8.0, 6.0],
    )

    assert output_path == tmp_path / "metrics.jsonl"
    assert output_path.exists()

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 3

    first = json.loads(lines[0])
    last = json.loads(lines[-1])

    assert first["epoch"] == 1
    assert first["train_loss"] == 100.0
    assert first["valid_rmse"] == 11.0

    assert last["epoch"] == 20
    assert last["train_loss"] == 25.0
    assert last["valid_mae"] == 5.0


def test_save_weight_history(tmp_path):
    from training.experiment_logging import save_weight_history

    output_path = save_weight_history(
        tmp_path,
        epochs=[1, 10],
        weight_history=[
            {"F01": 0.1, "F02": -0.2},
            {"F01": 0.3, "F02": -0.4},
        ],
        bias_history=[0.01, 0.02],
    )

    assert output_path == tmp_path / "weight_history.jsonl"
    assert output_path.exists()

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2

    first = json.loads(lines[0])
    last = json.loads(lines[-1])

    assert first["epoch"] == 1
    assert first["weights"]["F01"] == 0.1
    assert first["weights"]["F02"] == -0.2
    assert first["bias"] == 0.01

    assert last["epoch"] == 10
    assert last["weights"]["F01"] == 0.3
    assert last["bias"] == 0.02


def test_save_validation_predictions(tmp_path):
    from training.experiment_logging import save_validation_predictions

    rows = [
        {
            "game_id": 0,
            "position_index": 10,
            "target": 100.0,
            "prediction": 90.0,
        },
        {
            "game_id": 1,
            "position_index": 20,
            "target": -50.0,
            "prediction": -40.0,
        },
    ]

    output_path = save_validation_predictions(tmp_path, rows)

    assert output_path == tmp_path / "validation_predictions.jsonl"
    assert output_path.exists()

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert first["game_id"] == 0
    assert first["target"] == 100.0
    assert first["prediction"] == 90.0

    assert second["game_id"] == 1
    assert second["target"] == -50.0
