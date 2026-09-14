"""
training/trained_model.py

学習済み特徴量重みモデルの読み込みと予測を行う。

対応するモデル:
    training/linear_feature_weight_model.py
    scripts/train_feature_weight_model.py

主な機能:
    - final_model.json の読み込み
    - 特徴量名と重みの管理
    - 標準化統計量の適用
    - 特徴量ベクトルからの予測
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LoadedFeatureWeightModel:
    """
    学習済み特徴量重みモデル。

    Parameters
    ----------
    feature_names:
        モデルが想定する特徴量名の順序。
    weights:
        特徴量ごとの学習済み重み。
    bias:
        切片。
    means:
        標準化に使用した平均値。
    scales:
        標準化に使用した標準偏差。
    target_mode:
        学習時の教師値モード。
    """

    feature_names: tuple[str, ...]
    weights: np.ndarray
    bias: float
    means: np.ndarray | None = None
    scales: np.ndarray | None = None
    target_mode: str | None = None

    def __post_init__(self) -> None:
        """
        モデルの内部データを検証する。
        """
        feature_count = len(self.feature_names)

        if self.weights.ndim != 1:
            raise ValueError("weights must be a one-dimensional array")

        if len(self.weights) != feature_count:
            raise ValueError(
                "The number of weights must match feature_names: "
                f"{len(self.weights)} != {feature_count}"
            )

        if self.means is not None:
            if self.means.ndim != 1:
                raise ValueError("means must be a one-dimensional array")

            if len(self.means) != feature_count:
                raise ValueError(
                    "The number of means must match feature_names"
                )

        if self.scales is not None:
            if self.scales.ndim != 1:
                raise ValueError("scales must be a one-dimensional array")

            if len(self.scales) != feature_count:
                raise ValueError(
                    "The number of scales must match feature_names"
                )

            if np.any(self.scales <= 0.0):
                raise ValueError(
                    "All standardization scales must be positive"
                )

    @property
    def feature_count(self) -> int:
        """
        モデルが扱う特徴量数を返す。
        """
        return len(self.feature_names)

    @classmethod
    def from_json(cls, path: str | Path) -> "LoadedFeatureWeightModel":
        """
        JSON形式の学習済みモデルを読み込む。

        Parameters
        ----------
        path:
            final_model.jsonのパス。

        Returns
        -------
        LoadedFeatureWeightModel
            読み込まれたモデル。
        """
        model_path = Path(path)

        with model_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                f"Model JSON must contain an object: {model_path}"
            )

        feature_names = cls._load_feature_names(data)
        weights = cls._load_weights(data, feature_names)
        bias = cls._load_bias(data)

        means, scales = cls._load_standardization(data, feature_names)

        target_mode = data.get("target_mode")
        if target_mode is not None:
            target_mode = str(target_mode)

        return cls(
            feature_names=tuple(feature_names),
            weights=weights,
            bias=bias,
            means=means,
            scales=scales,
            target_mode=target_mode,
        )

    @staticmethod
    def _load_feature_names(data: Mapping[str, Any]) -> list[str]:
        """
        JSONから特徴量名を読み込む。
        """
        candidates = [
            data.get("feature_names"),
            data.get("features"),
        ]

        feature_names: Any = None

        for candidate in candidates:
            if candidate is not None:
                feature_names = candidate
                break

        if not isinstance(feature_names, list):
            raise ValueError(
                "Model JSON must contain 'feature_names' as a list"
            )

        if not feature_names:
            raise ValueError("feature_names must not be empty")

        result = [str(name) for name in feature_names]

        if len(set(result)) != len(result):
            raise ValueError("feature_names must not contain duplicates")

        return result

    @staticmethod
    def _load_weights(
        data: Mapping[str, Any],
        feature_names: Sequence[str],
    ) -> np.ndarray:
        """
        JSONから重みを読み込む。

        weightsは以下の形式に対応する。

        形式1:
            "weights": [0.1, 0.2, ...]

        形式2:
            "weights": {
                "F01": 0.1,
                "F02": 0.2
            }
        """
        raw_weights = data.get("weights")

        if raw_weights is None:
            raise ValueError("Model JSON must contain 'weights'")

        if isinstance(raw_weights, dict):
            try:
                values = [
                    float(raw_weights[name])
                    for name in feature_names
                ]
            except KeyError as error:
                raise ValueError(
                    f"Weight for feature is missing: {error.args[0]}"
                ) from error

            return np.asarray(values, dtype=np.float64)

        if isinstance(raw_weights, list):
            values = np.asarray(raw_weights, dtype=np.float64)

            if len(values) != len(feature_names):
                raise ValueError(
                    "The number of weights does not match feature_names"
                )

            return values

        raise ValueError(
            "'weights' must be either a list or an object"
        )

    @staticmethod
    def _load_bias(data: Mapping[str, Any]) -> float:
        """
        JSONからbiasを読み込む。
        """
        raw_bias = data.get("bias", data.get("intercept", 0.0))

        if isinstance(raw_bias, list):
            if len(raw_bias) != 1:
                raise ValueError(
                    "A list bias must contain exactly one value"
                )
            raw_bias = raw_bias[0]

        return float(raw_bias)

    @classmethod
    def _load_standardization(
        cls,
        data: Mapping[str, Any],
        feature_names: Sequence[str],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        標準化統計量を読み込む。

        対応形式:

        形式1:
            {
              "standardization": {
                "mean": {...},
                "std": {...}
              }
            }

        形式2:
            {
              "standardization": {
                "means": {...},
                "scales": {...}
              }
            }

        形式3:
            {
              "mean": [...],
              "std": [...]
            }

        標準化情報がない場合はNone, Noneを返す。
        """
        standardization = data.get("standardization")

        if isinstance(standardization, dict):
            raw_means = standardization.get(
                "mean",
                standardization.get("means"),
            )
            raw_scales = standardization.get(
                "std",
                standardization.get(
                    "stds",
                    standardization.get("scales"),
                ),
            )
        else:
            raw_means = data.get(
                "mean",
                data.get("means"),
            )
            raw_scales = data.get(
                "std",
                data.get(
                    "stds",
                    data.get("scales"),
                ),
            )

        if raw_means is None and raw_scales is None:
            return None, None

        if raw_means is None or raw_scales is None:
            raise ValueError(
                "Both mean and standard deviation are required "
                "for standardization"
            )

        means = cls._values_in_feature_order(
            raw_means,
            feature_names,
            "mean",
        )
        scales = cls._values_in_feature_order(
            raw_scales,
            feature_names,
            "std",
        )

        # 標準偏差が0の場合、学習時の実装によっては1に置換される。
        # ここでは0を許可せず、保存値の異常を明示する。
        if np.any(scales <= 0.0):
            raise ValueError(
                "Standardization standard deviations must be positive"
            )

        return means, scales

    @staticmethod
    def _values_in_feature_order(
        raw_values: Any,
        feature_names: Sequence[str],
        field_name: str,
    ) -> np.ndarray:
        """
        辞書または配列を特徴量順のNumPy配列へ変換する。
        """
        if isinstance(raw_values, dict):
            try:
                values = [
                    float(raw_values[name])
                    for name in feature_names
                ]
            except KeyError as error:
                raise ValueError(
                    f"{field_name} for feature is missing: "
                    f"{error.args[0]}"
                ) from error

            return np.asarray(values, dtype=np.float64)

        if isinstance(raw_values, list):
            values = np.asarray(raw_values, dtype=np.float64)

            if len(values) != len(feature_names):
                raise ValueError(
                    f"The number of {field_name} values does not "
                    "match feature_names"
                )

            return values

        raise ValueError(
            f"'{field_name}' must be either a list or an object"
        )

    def _prepare_vector(
        self,
        features: Mapping[str, float] | Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """
        入力特徴量をモデルの特徴量順に並べたベクトルへ変換する。
        """
        if isinstance(features, Mapping):
            missing_features = [
                name
                for name in self.feature_names
                if name not in features
            ]

            if missing_features:
                raise ValueError(
                    "Missing feature values: "
                    + ", ".join(missing_features)
                )

            vector = np.asarray(
                [
                    float(features[name])
                    for name in self.feature_names
                ],
                dtype=np.float64,
            )
        else:
            vector = np.asarray(features, dtype=np.float64)

            if vector.ndim != 1:
                raise ValueError(
                    "Feature vector must be one-dimensional"
                )

            if len(vector) != self.feature_count:
                raise ValueError(
                    "The number of input features does not match "
                    f"the model: {len(vector)} != {self.feature_count}"
                )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "Feature values must be finite"
            )

        return vector

    def transform_features(
        self,
        features: Mapping[str, float] | Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """
        入力特徴量をモデルの学習時と同じ形式に変換する。

        標準化統計量が保存されている場合:
            x' = (x - mean) / std

        標準化統計量がない場合:
            入力値をそのまま返す。
        """
        vector = self._prepare_vector(features)

        if self.means is None or self.scales is None:
            return vector

        return (vector - self.means) / self.scales

    def predict(
        self,
        features: Mapping[str, float] | Sequence[float] | np.ndarray,
    ) -> float:
        """
        特徴量から予測値を計算する。

        予測式:
            y = w・x + b
        """
        transformed = self.transform_features(features)

        prediction = float(
            np.dot(self.weights, transformed) + self.bias
        )

        return prediction

    def predict_batch(
        self,
        features: Sequence[
            Mapping[str, float] | Sequence[float] | np.ndarray
        ],
    ) -> np.ndarray:
        """
        複数の特徴量ベクトルをまとめて予測する。
        """
        predictions = [
            self.predict(feature_vector)
            for feature_vector in features
        ]

        return np.asarray(predictions, dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        """
        モデルの概要を辞書として返す。
        """
        return {
            "feature_count": self.feature_count,
            "feature_names": list(self.feature_names),
            "bias": self.bias,
            "has_standardization": (
                self.means is not None and self.scales is not None
            ),
            "target_mode": self.target_mode,
        }
