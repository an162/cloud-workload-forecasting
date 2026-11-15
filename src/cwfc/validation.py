from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass(frozen=True)
class Fold:
    """One expanding-window fold, addressed by position rather than by date."""

    index: int
    origin: int
    horizon: int

    @property
    def train_slice(self) -> slice:
        return slice(0, self.origin)

    @property
    def test_slice(self) -> slice:
        return slice(self.origin, self.origin + self.horizon)


def expanding_window_folds(
    n_obs: int, horizon: int, min_train: int, step: int = 1
) -> list[Fold]:
    """Expanding-window origins for a positionally indexed series.

    Fold `i` fits on `[0, origin)` and forecasts `[origin, origin + horizon)`.
    Origins start at `min_train` and advance by `step` while a full horizon
    still fits, so no fold is ever scored on a partial path and no training
    slice can touch its own test slice.

    Positional (not timestamp-based) addressing keeps this usable on the raw
    integer timestamp grid, and means a substituted dataset of any length
    yields however many folds it supports.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if min_train <= 0:
        raise ValueError("min_train must be positive")
    if step <= 0:
        raise ValueError("step must be positive")

    folds: list[Fold] = []
    origin = min_train
    while origin + horizon <= n_obs:
        folds.append(Fold(index=len(folds), origin=origin, horizon=horizon))
        origin += step
    return folds


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 288,
    min_train_size: int = 7 * 24 * 12,
    step: int | None = None,
) -> list[TimeSplit]:
    """Return expanding-window splits for evenly sampled time series.

    `horizon=288` is one day at 5-minute cadence.
    """
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    ts = pd.to_datetime(frame[timestamp_col], utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    n = len(frame)
    step = horizon if step is None else step
    out: list[TimeSplit] = []
    train_end_idx = min_train_size
    while train_end_idx + horizon <= n:
        train_end = ts.iloc[train_end_idx - 1]
        val_start = ts.iloc[train_end_idx]
        val_end = ts.iloc[train_end_idx + horizon - 1]
        out.append(TimeSplit(train_end, val_start, val_end))
        train_end_idx += step
    return out
