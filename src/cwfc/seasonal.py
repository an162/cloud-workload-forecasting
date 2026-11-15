from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WHITE_NOISE_Z = 1.96


def acf(x, max_lag: int) -> np.ndarray:
    """Sample autocorrelation at lags 1..max_lag (biased/standard estimator)."""
    v = np.asarray(x, dtype=float)
    if np.isnan(v).any():
        raise ValueError("acf requires a series without NaN")
    if max_lag < 1 or max_lag >= len(v):
        raise ValueError("max_lag must be in [1, len(x)-1]")
    centred = v - v.mean()
    denom = float((centred**2).sum())
    if denom == 0.0:
        return np.zeros(max_lag)
    return np.array(
        [float((centred[:-k] * centred[k:]).sum() / denom) for k in range(1, max_lag + 1)]
    )


def white_noise_band(n: int, z: float = WHITE_NOISE_Z) -> float:
    """Two-sided significance band for an ACF of a length-n white-noise series."""
    return float(z / np.sqrt(n))


def centered_moving_average(x, period: int) -> np.ndarray:
    """Centred moving average of one full period, NaN-padded at both ends.

    For an even period the standard 2xP weighting is used (half weight on the
    two end points) so the window is symmetric about each point; otherwise a
    plain P-point centred mean. Removing a full period of data is what makes
    the residual free of the seasonal component.
    """
    v = np.asarray(x, dtype=float)
    if period < 2:
        raise ValueError("period must be at least 2")
    if len(v) < period + 1:
        return np.full(len(v), np.nan)

    if period % 2 == 0:
        weights = np.ones(period + 1)
        weights[0] = weights[-1] = 0.5
    else:
        weights = np.ones(period)
    weights = weights / weights.sum()

    half = len(weights) // 2
    trend = np.full(len(v), np.nan)
    rolled = np.convolve(v, weights, mode="valid")
    trend[half : half + len(rolled)] = rolled
    return trend


@dataclass(frozen=True)
class PhaseProfile:
    """Additive seasonal effect per phase, indexed by position-mod-period."""

    period: int
    effect: np.ndarray
    counts: np.ndarray
    shrinkage: float

    @property
    def amplitude(self) -> float:
        return float(np.nanmax(self.effect) - np.nanmin(self.effect))


def phase_profile(x, period: int, shrinkage: float = 0.0) -> PhaseProfile:
    """Mean detrended deviation per phase, centred to sum to zero.

    The series is detrended by a centred one-period moving average, so the
    remainder holds seasonality plus noise but no level. Deviations are then
    averaged by phase = index % period.

    `shrinkage` pulls each phase effect toward zero by n/(n+shrinkage), where
    n is how many cycles contributed to that phase. With few cycles the raw
    phase means are noisy; shrinkage trades a little bias for variance. 0
    leaves the raw profile (used for description), >0 is for forecasting.
    """
    v = np.asarray(x, dtype=float)
    if period < 2:
        raise ValueError("period must be at least 2")
    if shrinkage < 0:
        raise ValueError("shrinkage must be non-negative")

    remainder = v - centered_moving_average(v, period)
    phases = np.arange(len(v)) % period

    effect = np.full(period, np.nan)
    counts = np.zeros(period, dtype="int64")
    for p in range(period):
        vals = remainder[phases == p]
        vals = vals[~np.isnan(vals)]
        counts[p] = len(vals)
        if len(vals):
            effect[p] = vals.mean()

    if shrinkage > 0:
        factor = counts / (counts + shrinkage)
        effect = effect * factor

    # Centre so the profile is purely redistributive and adds no level.
    effect = effect - np.nanmean(effect)
    return PhaseProfile(period=period, effect=effect, counts=counts, shrinkage=shrinkage)


def deseasonalize(x, profile: PhaseProfile, phase_offset: int = 0) -> np.ndarray:
    """Subtract the phase effect from each point of `x`."""
    v = np.asarray(x, dtype=float)
    phases = (np.arange(len(v)) + phase_offset) % profile.period
    return v - profile.effect[phases]


def min_length_for_period(period: int) -> int:
    """Shortest series on which `detect_period` can test the 2P harmonic.

    `detect_period` works on the differenced series (length n-1) and only
    examines lags up to (n-2)//2, so confirming the harmonic at lag 2P needs
    (n-2)//2 >= 2P, i.e. n >= 4P + 2. Callers size their first training window
    from this so no fold is asked a question the detector cannot answer.
    """
    p = int(period)
    if p < 2:
        raise ValueError("period must be at least 2")
    return 4 * p + 2


def detect_period(x, max_lag: int | None = None, z: float = WHITE_NOISE_Z) -> int | None:
    """Detect a seasonal period from the differenced series, or return None.

    Rule (PLAN 3): take the lag of the largest differenced-series ACF peak,
    and accept it only if the ACF at both that lag and twice that lag clear
    the white-noise band. Requiring the harmonic at 2P is what separates a
    real repeating cycle from a single spurious correlation.

    Seasonality is therefore detected per fold, never assumed.
    """
    v = np.asarray(x, dtype=float)
    diffs = np.diff(v)
    if len(diffs) < 4:
        return None

    ceiling = (len(diffs) - 1) // 2
    limit = ceiling if max_lag is None else min(int(max_lag), ceiling)
    if limit < 2:
        return None

    values = acf(diffs, limit)
    band = white_noise_band(len(diffs), z=z)

    candidate = int(np.argmax(values)) + 1
    if candidate < 2:
        return None
    if values[candidate - 1] <= band:
        return None
    harmonic = 2 * candidate
    if harmonic > limit or values[harmonic - 1] <= band:
        return None
    return candidate
