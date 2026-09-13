"""
学習ログ機能のテスト。
"""

import json
from pathlib import Path

import pytest

from training.logging_utils import (
    FEATURE_NAMES,
    create_run_directory,
    load_jsonl,
    log_metrics,
    log_weights,
    normalize_feature_weights,
    save_config,
)


def test_create_run_directory(tmp_path: Path) -> None:
    run_dir = create_run_directory(
        root_dir=tmp_path,
        run_name="experiment",
    )

    assert run_dir.exists()
    assert run_dir.is_dir()
    assert run_dir.name == "experiment"


def test_create_run_directory_adds_suffix(tmp_path: Path) -> None:
    first = create_run_directory(
        root_dir=tmp_path,
        run_name="experiment",
    )

    second = create_run_directory(
        root_dir=tmp_path,
        run_name="experiment",
    )

    assert first != second
    assert first.name == "experiment"
    assert second.name == "experiment_001"


def test_save_config(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, "config_test")

    output_path = save_config(
        run_dir,
        {
            "epochs": 10,
            "batch_size": 64,
            "feature_count": 40,
        },
    )

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    assert config["epochs"] == 10
    assert config["batch_size"] == 64
    assert config["feature_count"] == 40


def test_log_metrics(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, "metrics_test")

    output_path = log_metrics(
        run_dir,
        epoch=1,
        metrics={
            "train_loss": 0.8,
            "validation_loss": 0.9,
            "learning_rate": 0.001,
        },
    )

    assert output_path.exists()

    records = load_jsonl(output_path)

    assert len(records) == 1
    assert records[0]["epoch"] == 1
    assert records[0]["train_loss"] == pytest.approx(0.8)
    assert records[0]["validation_loss"] == pytest.approx(0.9)


def test_log_metrics_appends_records(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, "metrics_append_test")

    log_metrics(
        run_dir,
        epoch=1,
        metrics={"train_loss": 1.0},
    )
    log_metrics(
        run_dir,
        epoch=2,
        metrics={"train_loss": 0.5},
    )

    records = load_jsonl(run_dir / "metrics.jsonl")

    assert len(records) == 2
    assert records[0]["epoch"] == 1
    assert records[1]["epoch"] == 2


def test_normalize_feature_weights_from_sequence() -> None:
    weights = [0.1] * 40

    result = normalize_feature_weights(weights)

    assert len(result) == 40
    assert tuple(result.keys()) == FEATURE_NAMES
    assert result["F01"] == pytest.approx(0.1)
    assert result["F40"] == pytest.approx(0.1)


def test_normalize_feature_weights_from_mapping() -> None:
    result = normalize_feature_weights(
        {
            "F01": 0.5,
            "F02": 0.25,
        }
    )

    assert len(result) == 40
    assert result["F01"] == pytest.approx(0.5)
    assert result["F02"] == pytest.approx(0.25)
    assert result["F03"] == pytest.approx(0.0)


def test_normalize_feature_weights_rejects_invalid_length() -> None:
    with pytest.raises(ValueError, match="Expected 40 weights"):
        normalize_feature_weights([0.1] * 39)


def test_normalize_feature_weights_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown feature names"):
        normalize_feature_weights(
            {
                "F01": 0.1,
                "F99": 0.2,
            }
        )


def test_log_weights(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, "weights_test")

    weights = [float(index) for index in range(40)]

    output_path = log_weights(
        run_dir,
        epoch=3,
        weights=weights,
    )

    assert output_path.exists()

    records = load_jsonl(output_path)

    assert len(records) == 1
    assert records[0]["epoch"] == 3
    assert len(records[0]["weights"]) == 40
    assert records[0]["weights"]["F01"] == pytest.approx(0.0)
    assert records[0]["weights"]["F40"] == pytest.approx(39.0)


def test_log_weights_rejects_nan(tmp_path: Path) -> None:
    run_dir = create_run_directory(tmp_path, "weights_nan_test")

    weights = [0.0] * 40
    weights[0] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        log_weights(
            run_dir,
            epoch=1,
            weights=weights,
        )
