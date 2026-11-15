from __future__ import annotations

import numpy as np
import pandas as pd


def _as_train(train) -> np.ndarray:
    values = np.asarray(train, dtype=float)
    if values.size == 0:
        raise ValueError("train must not be empty")
    if np.isnan(values).any():
        raise ValueError("train must not contain NaN; handle gaps explicitly first")
    return values


def persistence(train, horizon: int, period: int | None = None) -> np.ndarray:
    """Flat forecast at the last observed value.

    Known weakness on a seasonal series: the last point sits at an arbitrary
    phase of the cycle, so the whole path inherits that phase's offset.
    """
    return np.repeat(_as_train(train)[-1], horizon)


def last_cycle_mean(train, horizon: int, period: int | None = None) -> np.ndarray:
    """Flat forecast at the mean of the last full cycle.

    Averaging exactly one period cancels the seasonal component, which makes
    this the fair "the level persists" baseline rather than persistence.
    Without a detected period it falls back to the mean of all history.
    """
    values = _as_train(train)
    window = len(values) if period is None else min(int(period), len(values))
    return np.repeat(values[-window:].mean(), horizon)


def seasonal_naive(train, horizon: int, period: int | None = None) -> np.ndarray:
    """Repeat the last `period` observations forward.

    Falls back to persistence when no period was detected, so the harness can
    run unchanged on a non-seasonal series.
    """
    values = _as_train(train)
    if period is None or int(period) < 1:
        return persistence(values, horizon)
    p = int(period)
    if len(values) < p:
        raise ValueError("not enough history for a seasonal-naive forecast")
    tail = values[-p:]
    return np.array([tail[h % p] for h in range(horizon)])


def train_mean(train, horizon: int, period: int | None = None) -> np.ndarray:
    """Flat forecast at the mean of all training history.

    Reference line for the predictability curve: a model that cannot beat this
    has extracted nothing time-dependent from the series.
    """
    return np.repeat(_as_train(train).mean(), horizon)


def persistence_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    if train.empty:
        raise ValueError("train must not be empty")
    last_value = float(train.iloc[-1][target])
    return pd.Series(last_value, index=template.index, dtype=float)


def seasonal_naive_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, season_length: int) -> pd.Series:
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    values = train[target].astype(float).to_numpy()
    if len(values) < season_length:
        raise ValueError("not enough history for seasonal-naive forecast")
    forecast = [float(values[-season_length + (i % season_length)]) for i in range(len(template))]
    return pd.Series(forecast, index=template.index, dtype=float)
