"""
tests/test_trained_model.py

学習済み特徴量重みモデルの読み込み・予測テスト。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from training.trained_model import LoadedFeatureWeightModel


def write_json(path: Path, data: dict) -> None:
    """
    テスト用JSONを書き込む。
    """
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_load_model_with_weight_dictionary(tmp_path: Path) -> None:
    """
    重みが辞書形式のモデルを読み込めることを確認する。
    """
    model_path = tmp_path / "final_model.json"

    write_json(
        model_path,
        {
            "feature_names": ["F01", "F02"],
            "weights": {
                "F01": 2.0,
                "F02": -1.0,
            },
            "bias": 0.5,
        },
    )

    model = LoadedFeatureWeightModel.from_json(model_path)

    assert model.feature_names == ("F01", "F02")
    assert model.feature_count == 2
    assert model.bias == pytest.approx(0.5)
    np.testing.assert_allclose(
        model.weights,
        np.asarray([2.0, -1.0]),
    )


def test_load_model_with_weight_list(tmp_path: Path) -> None:
    """
    重みが配列形式のモデルを読み込めることを確認する。
    """
    model_path = tmp_path / "final_model.json"

    write_json(
        model_path,
        {
            "feature_names": ["F01", "F02"],
            "weights": [2.0, -1.0],
            "bias": 0.5,
        },
    )

    model = LoadedFeatureWeightModel.from_json(model_path)

    np.testing.assert_allclose(
        model.weights,
        np.asarray([2.0, -1.0]),
    )


def test_predict_from_mapping(tmp_path: Path) -> None:
    """
    特徴量名をキーとする辞書から予測できることを確認する。
    """
    model_path = tmp_path / "final_model.json"

    write_json(
        model_path,
        {
            "feature_names": ["F01", "F02"],
            "weights": {
                "F01": 2.0,
                "F02": -1.0,
            },
            "bias": 0.5,
        },
    )

    model = LoadedFeatureWeightModel.from_json(model_path)

    prediction = model.predict(
        {
            "F01": 3.0,
            "F02": 2.0,
        }
    )

    # 2 * 3 + (-1) * 2 + 0.5 = 4.5
    assert prediction == pytest.approx(4.5)


def test_predict_from_vector_uses_feature_order(
    tmp_path: Path,
) -> None:
    """
    配列入力がfeature_namesの順序で計算されることを確認する。
    """
    model_path = tmp_path / "final_model.json"

    write_json(
        model_path,
        {
            "feature_names": ["F01", "F02"],
            "weights": [2.0, -1.0],
            "bias": 0.5,
        },
    )

    model = LoadedFeatureWeightModel.from_json(model_path)

    prediction = model.predict([3.0, 2.0])

    assert prediction == pytest.approx(4.5)


def test_predict_applies_standardization(
    tmp_path: Path,
) -> None:
    """
    標準化統計量が予測前に適用されることを確認する。
    """
    model_path = tmp_path / "final_model.json"

    write_json(
        model_path,
        {
            "feature_names": ["F01", "F02"],
            "weights": [2.0, -1.0],
            "bias": 0.5,
            "standardization": {
                "mean": [1.0, 2.0],
                "std": [2.0, 4.0],
            },
        },
    )

    model = LoadedFeatureWeightModel.from_json(model_path)

    # 標準化後:
    # F01: (5 - 1) / 2 = 2
    # F02: (6 - 2) / 4 = 1
    #
    # 2 * 2 - 1 * 1 + 0.5 = 3.5
    prediction = model.predict([5.0, 6.0])

    assert prediction == pytest.approx(3.5)


def test_predict_batch(tmp_path: Path) -> None:
    """
    複数入力をまとめて予測できることを確認する。
    """
    model_path = tmp_path / "final_model.json"

    write_json(
        model_path,
        {
            "feature_names": ["F01", "F02"],
            "weights": [2.0, -1.0],
            "bias": 0.5,
        },
    )

    model = LoadedFeatureWeightModel.from_json(model_path)

    predictions = model.predict_batch(
        [
            [3.0, 2.0],
            [1.0, 4.0],
        ]
    )

    np.testing.assert_allclose(
        predictions,
        np.asarray([4.5, -1.5]),
    )


def test_missing_feature_raises_error(tmp_path: Path) -> None:
    """
    必要な特徴量が不足している場合にエラーになることを確認する。
    """
    model_path = tmp_path / "final_model.json"

    write_json(
        model_path,
        {
            "feature_names": ["F01", "F02"],
            "weights": [2.0, -1.0],
            "bias": 0.5,
        },
    )

    model = LoadedFeatureWeightModel.from_json(model_path)

    with pytest.raises(ValueError, match="Missing feature values"):
        model.predict({"F01": 3.0})


def test_invalid_vector_length_raises_error(
    tmp_path: Path,
) -> None:
    """
    入力ベクトルの次元数が異なる場合にエラーになることを確認する。
    """
    model_path = tmp_path / "final_model.json"

    write_json(
        model_path,
        {
            "feature_names": ["F01", "F02"],
            "weights": [2.0, -1.0],
            "bias": 0.5,
        },
    )

    model = LoadedFeatureWeightModel.from_json(model_path)

    with pytest.raises(ValueError, match="number of input features"):
        model.predict([1.0])


def test_invalid_feature_names_raise_error(tmp_path: Path) -> None:
    """
    feature_namesが存在しない場合にエラーになることを確認する。
    """
    model_path = tmp_path / "final_model.json"

    write_json(
        model_path,
        {
            "weights": [1.0, 2.0],
            "bias": 0.0,
        },
    )

    with pytest.raises(ValueError, match="feature_names"):
        LoadedFeatureWeightModel.from_json(model_path)


def test_non_finite_feature_raises_error(tmp_path: Path) -> None:
    """
    NaNや無限大の入力を拒否することを確認する。
    """
    model_path = tmp_path / "final_model.json"

    write_json(
        model_path,
        {
            "feature_names": ["F01", "F02"],
            "weights": [2.0, -1.0],
            "bias": 0.5,
        },
    )

    model = LoadedFeatureWeightModel.from_json(model_path)

    with pytest.raises(ValueError, match="finite"):
        model.predict([float("nan"), 1.0])
