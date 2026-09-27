from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass
class StandardizationStats:
    """
    特徴量の標準化に使用する統計量。
    """

    mean: np.ndarray
    std: np.ndarray


@dataclass
class LinearModel:
    """
    線形回帰モデル。

    予測式:
        y_hat = X @ weights + bias
    """

    weights: np.ndarray
    bias: float
    feature_names: tuple[str, ...]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        標準化済み特徴量Xから予測値を計算する。
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                f"X must be 2-dimensional, got shape={X.shape}"
            )

        if X.shape[1] != len(self.weights):
            raise ValueError(
                "Feature dimension mismatch: "
                f"X has {X.shape[1]} features, "
                f"model expects {len(self.weights)}"
            )

        return X @ self.weights + self.bias


@dataclass
class TrainingHistory:
    """
    学習過程の履歴。

    Attributes
    ----------
    epochs:
        記録したエポック番号。
    train_loss:
        学習データに対するMSE損失。
    valid_loss:
        検証データに対するMSE損失。
    train_mae:
        学習データに対するMAE。
    valid_mae:
        検証データに対するMAE。
    train_rmse:
        学習データに対するRMSE。
    valid_rmse:
        検証データに対するRMSE。
    weight_history:
        各記録エポックにおける特徴量重み。
    bias_history:
        各記録エポックにおけるバイアス。
    """

    epochs: list[int]
    train_loss: list[float]
    valid_loss: list[float]
    train_mae: list[float]
    valid_mae: list[float]
    train_rmse: list[float]
    valid_rmse: list[float]
    weight_history: list[dict[str, float]] = field(
        default_factory=list
    )
    bias_history: list[float] = field(
        default_factory=list
    )


def fit_standardization(
    X: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> StandardizationStats:
    """
    学習データから標準化統計量を計算する。

    標準化:
        X_scaled = (X - mean) / std

    標準偏差がほぼ0の特徴量はstd=1として扱う。
    """

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(
            f"X must be 2-dimensional, got shape={X.shape}"
        )

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # 定数特徴量によるゼロ除算を防止する。
    std = np.where(std < epsilon, 1.0, std)

    return StandardizationStats(
        mean=mean,
        std=std,
    )


def apply_standardization(
    X: np.ndarray,
    stats: StandardizationStats,
) -> np.ndarray:
    """
    指定された統計量で特徴量を標準化する。
    """

    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(
            f"X must be 2-dimensional, got shape={X.shape}"
        )

    if X.shape[1] != len(stats.mean):
        raise ValueError(
            "Feature dimension mismatch between X and statistics"
        )

    return (X - stats.mean) / stats.std


def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    平均二乗誤差。
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    平均絶対誤差。
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    平均平方二乗誤差。
    """

    return float(
        np.sqrt(mean_squared_error(y_true, y_pred))
    )


def pearson_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float | None:
    """
    Pearson相関係数。

    片方の分散が0の場合は計算不能としてNoneを返す。
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) < 2:
        return None

    true_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))

    if true_std == 0.0 or pred_std == 0.0:
        return None

    value = np.corrcoef(y_true, y_pred)[0, 1]

    if not np.isfinite(value):
        return None

    return float(value)


def rankdata(values: np.ndarray) -> np.ndarray:
    """
    scipyに依存しない順位変換。

    同順位がある場合は、同順位範囲の平均順位を割り当てる。
    順位は1始まり。
    """

    values = np.asarray(values, dtype=np.float64)

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]

    ranks_sorted = np.empty(len(values), dtype=np.float64)

    start = 0

    while start < len(values):
        end = start + 1

        while (
            end < len(values)
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1

        # 1始まりの順位の平均
        average_rank = (start + 1 + end) / 2.0
        ranks_sorted[start:end] = average_rank

        start = end

    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = ranks_sorted

    return ranks


def spearman_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float | None:
    """
    Spearman順位相関係数。
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) < 2:
        return None

    true_ranks = rankdata(y_true)
    pred_ranks = rankdata(y_pred)

    return pearson_correlation(true_ranks, pred_ranks)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int | None]:
    """
    予測結果を複数の指標で評価する。
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return {
        "count": int(len(y_true)),
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "pearson": pearson_correlation(y_true, y_pred),
        "spearman": spearman_correlation(y_true, y_pred),
    }


def clip_targets(
    y: np.ndarray,
    limit: float | None,
) -> np.ndarray:
    """
    教師値を指定範囲にクリッピングする。

    limit=Noneの場合は変更しない。
    """

    y = np.asarray(y, dtype=np.float64)

    if limit is None:
        return y.copy()

    if limit <= 0:
        raise ValueError("limit must be positive or None")

    return np.clip(y, -limit, limit)


def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_names: Sequence[str],
    *,
    epochs: int = 500,
    learning_rate: float = 0.01,
    l2_strength: float = 1e-3,
    log_interval: int = 10,
    seed: int = 42,
) -> tuple[LinearModel, TrainingHistory]:
    """
    標準化済み特徴量に対して線形回帰を学習する。

    損失関数:
        MSE + L2正則化

    biasにはL2正則化を適用しない。
    """

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_valid = np.asarray(X_valid, dtype=np.float64)
    y_valid = np.asarray(y_valid, dtype=np.float64)

    if X_train.ndim != 2:
        raise ValueError("X_train must be 2-dimensional")

    if X_valid.ndim != 2:
        raise ValueError("X_valid must be 2-dimensional")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train length mismatch")

    if len(X_valid) != len(y_valid):
        raise ValueError("X_valid and y_valid length mismatch")

    if X_train.shape[1] != X_valid.shape[1]:
        raise ValueError(
            "Train and validation feature dimensions differ"
        )

    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            "Feature names and X dimension mismatch"
        )

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if l2_strength < 0:
        raise ValueError("l2_strength must be non-negative")

    rng = np.random.default_rng(seed)

    # 小さい乱数で初期化する。
    weights = rng.normal(
        loc=0.0,
        scale=0.01,
        size=X_train.shape[1],
    )

    bias = float(np.mean(y_train))

    history = TrainingHistory(
        epochs=[],
        train_loss=[],
        valid_loss=[],
        train_mae=[],
        valid_mae=[],
        train_rmse=[],
        valid_rmse=[],
        weight_history=[],
        bias_history=[],
    )

    sample_count = float(len(X_train))

    for epoch in range(1, epochs + 1):
        train_pred = X_train @ weights + bias
        errors = train_pred - y_train

        # MSEの勾配
        grad_weights = (
            2.0 / sample_count
        ) * (X_train.T @ errors)

        grad_bias = float(
            2.0 * np.mean(errors)
        )

        # L2正則化。biasには適用しない。
        grad_weights += 2.0 * l2_strength * weights

        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias

        should_log = (
            epoch == 1
            or epoch == epochs
            or epoch % log_interval == 0
        )

        if should_log:
            train_pred = X_train @ weights + bias
            valid_pred = X_valid @ weights + bias

            train_mse = mean_squared_error(
                y_train,
                train_pred,
            )
            valid_mse = mean_squared_error(
                y_valid,
                valid_pred,
            )

            history.epochs.append(epoch)
            history.train_loss.append(train_mse)
            history.valid_loss.append(valid_mse)
            history.train_mae.append(
                mean_absolute_error(y_train, train_pred)
            )
            history.valid_mae.append(
                mean_absolute_error(y_valid, valid_pred)
            )
            history.train_rmse.append(
                root_mean_squared_error(y_train, train_pred)
            )
            history.valid_rmse.append(
                root_mean_squared_error(y_valid, valid_pred)
            )
            history.weight_history.append(
                {
                    str(name): float(weight)
                 for name, weight in zip(feature_names, weights)
                }
            )
            history.bias_history.append(float(bias))

    model = LinearModel(
        weights=weights,
        bias=bias,
        feature_names=tuple(feature_names),
    )

    return model, history
