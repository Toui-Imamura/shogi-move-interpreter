from __future__ import annotations

import numpy as np

from training.linear_feature_weight_model import (
    apply_standardization,
    clip_targets,
    evaluate_predictions,
    fit_standardization,
    pearson_correlation,
    spearman_correlation,
    train_linear_regression,
)


def test_standardization_has_zero_mean_and_unit_std():
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    stats = fit_standardization(X)
    X_scaled = apply_standardization(X, stats)

    assert np.allclose(
        np.mean(X_scaled, axis=0),
        np.zeros(2),
    )

    assert np.allclose(
        np.std(X_scaled, axis=0),
        np.ones(2),
    )


def test_standardization_handles_constant_feature():
    X = np.array(
        [
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
        ]
    )

    stats = fit_standardization(X)
    X_scaled = apply_standardization(X, stats)

    assert np.all(np.isfinite(X_scaled))
    assert np.allclose(X_scaled[:, 1], 0.0)


def test_clip_targets():
    y = np.array([-5000.0, -100.0, 0.0, 100.0, 5000.0])

    clipped = clip_targets(y, 2000.0)

    assert np.allclose(
        clipped,
        np.array([-2000.0, -100.0, 0.0, 100.0, 2000.0]),
    )


def test_clip_targets_none():
    y = np.array([-5000.0, 100.0, 5000.0])

    clipped = clip_targets(y, None)

    assert np.array_equal(clipped, y)


def test_correlation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 4.0, 6.0, 8.0])

    assert np.isclose(
        pearson_correlation(y_true, y_pred),
        1.0,
    )

    assert np.isclose(
        spearman_correlation(y_true, y_pred),
        1.0,
    )


def test_evaluate_predictions():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 3.0, 2.0])

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics["count"] == 3
    assert np.isclose(metrics["mae"], 2.0 / 3.0)
    assert np.isclose(metrics["rmse"], np.sqrt(2.0 / 3.0))


def test_train_linear_regression_learns_simple_relation():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(100, 2))
    true_weights = np.array([3.0, -2.0])
    true_bias = 1.5

    y = X @ true_weights + true_bias

    X_train = X[:80]
    y_train = y[:80]
    X_valid = X[80:]
    y_valid = y[80:]

    model, history = train_linear_regression(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        feature_names=("F01", "F02"),
        epochs=1000,
        learning_rate=0.05,
        l2_strength=0.0,
        log_interval=100,
        seed=42,
    )

    predictions = model.predict(X_valid)

    assert len(history.epochs) > 0
    assert np.mean((predictions - y_valid) ** 2) < 0.01


def test_train_linear_regression_records_weight_history():
    import numpy as np

    from training.linear_feature_weight_model import (
        train_linear_regression,
    )

    X_train = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    y_train = np.array(
        [0.0, 1.0, 2.0, 3.0],
        dtype=float,
    )

    X_valid = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    y_valid = np.array(
        [0.0, 3.0],
        dtype=float,
    )

    feature_names = ["F01", "F02"]

    model, history = train_linear_regression(
        X_train,
        y_train,
        X_valid,
        y_valid,
        feature_names,
        epochs=5,
        learning_rate=0.05,
        l2_strength=0.0,
        log_interval=1,
        seed=42,
    )

    assert len(history.epochs) == 5
    assert len(history.weight_history) == 5
    assert len(history.bias_history) == 5

    assert history.epochs == [1, 2, 3, 4, 5]

    assert set(history.weight_history[0].keys()) == {
        "F01",
        "F02",
    }

    assert all(
        isinstance(value, float)
        for value in history.weight_history[0].values()
    )

    assert all(
        isinstance(value, float)
        for value in history.bias_history
    )

    assert history.weight_history[0] != history.weight_history[-1]
    assert history.bias_history[0] != history.bias_history[-1]
