from __future__ import annotations

import warnings

import numpy as np

from cwfc import baselines
from cwfc.seasonal import acf, deseasonalize, phase_profile

DEFAULT_SHRINKAGE = 1.0


def _checked(train) -> np.ndarray:
    values = np.asarray(train, dtype=float)
    if values.size == 0:
        raise ValueError("train must not be empty")
    if np.isnan(values).any():
        raise ValueError("train must not contain NaN; handle gaps explicitly first")
    return values


def _profile_and_level_series(
    train: np.ndarray, period: int | None, shrinkage: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (phase effect by phase, deseasonalized series).

    With no detected period the phase effect is identically zero, so every
    model below degrades to its own non-seasonal analogue rather than failing.
    """
    if period is None:
        return np.zeros(1), train.copy()
    prof = phase_profile(train, int(period), shrinkage=shrinkage)
    effect = np.nan_to_num(prof.effect, nan=0.0)
    return effect, deseasonalize(train, prof)


def _effect_at(effect: np.ndarray, index: int, period: int | None) -> float:
    if period is None:
        return 0.0
    return float(effect[index % int(period)])


def _cycle_window(train: np.ndarray, period: int | None) -> int:
    return len(train) if period is None else min(int(period), len(train))


def profile_last_cycle(
    train, horizon: int, period: int | None = None, shrinkage: float = DEFAULT_SHRINKAGE
) -> np.ndarray:
    """PLAN candidate 5: last-cycle level + phase profile.

    The level is the mean of the last full cycle of the *deseasonalized*
    series, so it is not contaminated by whichever phase the series happens to
    end on; the phase effect is then added back for each forecast position.
    """
    values = _checked(train)
    effect, deseas = _profile_and_level_series(values, period, shrinkage)
    level = float(deseas[-_cycle_window(values, period) :].mean())
    n = len(values)
    return np.array(
        [level + _effect_at(effect, n + h, period) for h in range(horizon)], dtype=float
    )


def profile_train_mean(
    train, horizon: int, period: int | None = None, shrinkage: float = DEFAULT_SHRINKAGE
) -> np.ndarray:
    """New candidate (a): full-training-mean level + phase profile.

    Motivated by the baseline result that the training mean beats every
    recency-anchored baseline over a 24-step path: this keeps that level and
    adds back only the within-hour shape.
    """
    values = _checked(train)
    effect, deseas = _profile_and_level_series(values, period, shrinkage)
    level = float(deseas.mean())
    n = len(values)
    return np.array(
        [level + _effect_at(effect, n + h, period) for h in range(horizon)], dtype=float
    )


def ar1_phi(deseasonalized) -> float:
    """Lag-1 autocorrelation of the deseasonalized series, clipped to [0, 1).

    DIAGNOSTIC ONLY - no model uses this. It is retained so the evidence
    behind DECISIONS entry 18 stays reproducible: the fitted reversion rate
    is unstable across folds (~0.39 in the earliest folds, ~0.85 once the
    steps 53-72 bump enters the training window), which is why the AR(1)
    reversion term was dropped rather than selected (entry 20).

    Estimating it after removing the seasonal profile matters: on the raw
    series the hourly sawtooth drags lag-1 down and understates persistence.
    """
    values = np.asarray(deseasonalized, dtype=float)
    if len(values) < 3:
        return 0.0
    return float(np.clip(acf(values, 1)[0], 0.0, 0.999))


def holt_winters(train, horizon: int, period: int | None = None) -> np.ndarray:
    """PLAN candidate 7: additive Holt-Winters with damped trend.

    Falls back to persistence with a warning if statsmodels is unavailable,
    the history is too short for two full seasons, or the fit fails to
    converge to finite forecasts (PLAN 2's too-little-data rule).
    """
    values = _checked(train)
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:  # pragma: no cover - environment dependent
        warnings.warn("statsmodels unavailable; falling back to persistence", RuntimeWarning)
        return baselines.persistence(values, horizon)

    use_seasonal = period is not None and len(values) >= 2 * int(period)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ExponentialSmoothing(
                values,
                trend="add",
                damped_trend=True,
                seasonal="add" if use_seasonal else None,
                seasonal_periods=int(period) if use_seasonal else None,
                initialization_method="estimated",
            ).fit(optimized=True)
            pred = np.asarray(fitted.forecast(horizon), dtype=float)
        if pred.shape != (horizon,) or not np.all(np.isfinite(pred)):
            raise ValueError("non-finite Holt-Winters forecast")
        return pred
    except Exception as exc:  # noqa: BLE001 - any fit failure degrades gracefully
        warnings.warn(f"Holt-Winters fit failed ({exc}); using persistence", RuntimeWarning)
        return baselines.persistence(values, horizon)


CANDIDATES = {
    "profile_last_cycle": profile_last_cycle,
    "profile_train_mean": profile_train_mean,
    "holt_winters": holt_winters,
}
