"""
interpreter/feature_weight_inference.py

局面特徴量と学習済み特徴量重みモデルを接続する。

役割:
1. 局面から得られた全特徴量を受け取る
2. 学習済みモデルが要求する特徴量だけを抽出する
3. 学習済みモデルで予測値を計算する

このモジュールは、cshogiそのものの特徴量計算を担当しない。
特徴量計算とモデル推論を分離することで、テストしやすくする。
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from training.trained_model import LoadedFeatureWeightModel


class FeatureWeightInference:
    """
    特徴量辞書と学習済みモデルを接続する推論クラス。

    Parameters
    ----------
    model:
        読み込み済みのLoadedFeatureWeightModel。
    """

    def __init__(self, model: LoadedFeatureWeightModel) -> None:
        self.model = model

    @classmethod
    def from_json(cls, model_path: str | Path) -> "FeatureWeightInference":
        """
        JSON形式の学習済みモデルから推論器を生成する。
        """
        model = LoadedFeatureWeightModel.from_json(model_path)
        return cls(model)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """
        モデルが要求する特徴量名を返す。
        """
        return self.model.feature_names

    def select_model_features(
        self,
        features: Mapping[str, float],
        *,
        default: float = 0.0,
        strict: bool = False,
    ) -> dict[str, float]:
        """
        全特徴量からモデルが使用する特徴量だけを抽出する。

        Parameters
        ----------
        features:
            F01～F40などの特徴量辞書。
        default:
            入力辞書に存在しない特徴量へ設定する値。
        strict:
            Trueの場合、必要な特徴量が不足していれば例外を送出する。

        Returns
        -------
        dict[str, float]
            モデルのfeature_namesと同じ順序の特徴量辞書。
        """
        missing = [
            name
            for name in self.model.feature_names
            if name not in features
        ]

        if strict and missing:
            raise KeyError(
                "Missing model features: "
                + ", ".join(missing)
            )

        return {
            name: float(features.get(name, default))
            for name in self.model.feature_names
        }

    def predict(
        self,
        features: Mapping[str, float],
        *,
        default: float = 0.0,
        strict: bool = False,
    ) -> float:
        """
        特徴量辞書から予測値を計算する。
        """
        selected = self.select_model_features(
            features,
            default=default,
            strict=strict,
        )
        return float(self.model.predict(selected))

    def predict_with_details(
        self,
        features: Mapping[str, float],
        *,
        default: float = 0.0,
        strict: bool = False,
    ) -> dict[str, Any]:
        """
        予測値と入力特徴量、各特徴量の寄与をまとめて返す。

        Returns
        -------
        dict
            prediction:
                予測値
            bias:
                モデルのbias
            features:
                モデル入力に使用した特徴量
            contributions:
                各特徴量の重み付き寄与
        """
        selected = self.select_model_features(
            features,
            default=default,
            strict=strict,
        )

        contributions = {
            name: float(selected[name] * self.model.weights[index])
            for index, name in enumerate(self.model.feature_names)
        }

        prediction = float(
            self.model.bias + sum(contributions.values())
        )

        return {
            "prediction": prediction,
            "bias": float(self.model.bias),
            "features": selected,
            "contributions": contributions,
        }
