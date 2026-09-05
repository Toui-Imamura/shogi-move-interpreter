"""
F01～F40 統合パイプライン

現在局面と未来局面から特徴量変化をまとめ、
F34～F40の統合特徴量へ渡すための基盤。

この段階では、各特徴量の計算器を外部から登録できる
設計にしている。

理由:
    F01～F33には、それぞれ異なる引数・戻り値を持つ
    実装が存在するため、統合層から各特徴量の内部実装を
    直接推測して呼び出さない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


FeatureExtractor = Callable[[Any], float]


@dataclass(frozen=True)
class FeatureValuePair:
    """現在局面と未来局面における1特徴量の値。"""

    current: float
    future: float

    @property
    def delta(self) -> float:
        """未来局面 - 現在局面。"""
        return self.future - self.current


@dataclass(frozen=True)
class FeaturePipelineResult:
    """F01～F33の計算結果。"""

    feature_values: Mapping[str, FeatureValuePair]

    @property
    def deltas(self) -> dict[str, float]:
        """特徴量変化量を辞書として取得する。"""
        return {
            name: value.delta
            for name, value in self.feature_values.items()
        }

    def get(
        self,
        feature_name: str,
    ) -> FeatureValuePair | None:
        """指定した特徴量の現在値・未来値を取得する。"""
        return self.feature_values.get(feature_name)


def compute_feature_deltas(
    current: Any,
    future: Any,
    extractors: Mapping[str, FeatureExtractor],
) -> FeaturePipelineResult:
    """
    現在局面と未来局面から各特徴量の変化量を計算する。

    Parameters
    ----------
    current:
        現在局面 S0。

    future:
        未来局面 Sv。

    extractors:
        特徴量名から特徴量計算関数へのマッピング。

        例:
            {
                "F01": f01_extractor,
                "F02": f02_extractor,
            }

        各extractorは局面を1つ受け取り、
        その特徴量の数値を返す。

    Returns
    -------
    FeaturePipelineResult
        現在値、未来値、変化量を保持する結果。
    """

    if current is None:
        raise ValueError("current position must not be None")

    if future is None:
        raise ValueError("future position must not be None")

    feature_values: dict[str, FeatureValuePair] = {}

    for feature_name, extractor in extractors.items():
        current_value = float(extractor(current))
        future_value = float(extractor(future))

        feature_values[feature_name] = FeatureValuePair(
            current=current_value,
            future=future_value,
        )

    return FeaturePipelineResult(
        feature_values=feature_values,
    )