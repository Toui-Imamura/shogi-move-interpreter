"""
scripts/predict_with_trained_model.py

学習済み特徴量重みモデルを読み込み、
指定された特徴量から予測値を計算する。

使用例:

python scripts/predict_with_trained_model.py \
  data/training_logs/linear_feature_weight_wcsc36_10games_logged/final_model.json \
  --features F01=1.0 F02=0.0 F03=0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.trained_model import LoadedFeatureWeightModel  # noqa: E402


def parse_feature_values(
    values: list[str],
) -> dict[str, float]:
    """
    F01=1.0形式の引数を辞書へ変換する。
    """
    features: dict[str, float] = {}

    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Feature must use NAME=VALUE format: {value}"
            )

        name, raw_number = value.split("=", 1)
        name = name.strip()

        if not name:
            raise ValueError(
                f"Feature name must not be empty: {value}"
            )

        if name in features:
            raise ValueError(
                f"Duplicate feature: {name}"
            )

        try:
            features[name] = float(raw_number)
        except ValueError as error:
            raise ValueError(
                f"Feature value must be numeric: {value}"
            ) from error

    return features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict with a trained feature weight model."
    )

    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to final_model.json",
    )

    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Feature values in NAME=VALUE format.",
    )

    args = parser.parse_args()

    model = LoadedFeatureWeightModel.from_json(args.model_path)
    features = parse_feature_values(args.features)

    prediction = model.predict(features)

    print("Model:", args.model_path)
    print("Feature count:", model.feature_count)
    print("Provided features:", len(features))
    print("Prediction:", prediction)


if __name__ == "__main__":
    main()
