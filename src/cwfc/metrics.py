from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    denom = np.maximum(np.abs(a) + np.abs(b), eps)
    return float(np.mean(2.0 * np.abs(a - b) / denom) * 100.0)


def bias(y_true, y_pred) -> float:
    """Mean signed error (pred - true). Positive means over-forecasting."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(b - a))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: sum|e| / sum|y|, as a percentage.

    Used instead of MAPE because it divides by the aggregate level once rather
    than by each observation, so a single near-zero actual cannot blow the
    metric up. Returns NaN when the actuals sum to zero.
    """
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    denom = float(np.abs(a).sum())
    if denom == 0.0:
        return float("nan")
    return float(np.abs(a - b).sum() / denom * 100.0)


def naive_scale(train, eps: float = 1e-9) -> float:
    """In-sample one-step naive MAE, the MASE denominator.

    Returns NaN when the series is (near) constant, because a flat series
    gives the naive forecaster zero error and MASE is then undefined rather
    than infinite. The constant `active_vm_count` target hits this case.
    """
    values = np.asarray(train, dtype=float)
    if len(values) < 2:
        return float("nan")
    scale = float(np.abs(np.diff(values)).mean())
    return float("nan") if scale < eps else scale


def mase(y_true, y_pred, scale: float) -> float:
    """Mean absolute scaled error against a provided in-fold naive scale."""
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(mae(y_true, y_pred) / scale)


def skill(y_true, y_pred, baseline_pred) -> float:
    """Relative MAE improvement over a baseline: 1 - MAE_model / MAE_base.

    1.0 is perfect, 0.0 matches the baseline, negative is worse than it.
    """
    base = mae(y_true, baseline_pred)
    if base == 0:
        return float("nan")
    return float(1.0 - mae(y_true, y_pred) / base)


def normalized_mae(y_true, y_pred, baseline_pred) -> float:
    model_mae = mae(y_true, y_pred)
    baseline_mae = mae(y_true, baseline_pred)
    if baseline_mae == 0:
        return 0.0 if model_mae == 0 else float("inf")
    return float(model_mae / baseline_mae)
