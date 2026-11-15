"""Seasonality detection and profile mechanics (PLAN 8)."""

import numpy as np
import pytest

from cwfc.seasonal import (
    acf,
    centered_moving_average,
    deseasonalize,
    detect_period,
    min_length_for_period,
    phase_profile,
    white_noise_band,
)

PERIOD = 12


def sawtooth(n=120, period=PERIOD, amplitude=1.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    phase = np.arange(n) % period
    series = 10.0 + amplitude * np.where(phase == 4, -2.0, np.where(phase == 5, 1.0, 0.0))
    return series + rng.normal(0, noise, n)


def test_detects_the_planted_period():
    assert detect_period(sawtooth()) == PERIOD
    assert detect_period(sawtooth(noise=0.05)) == PERIOD


def test_finds_no_period_in_noise():
    rng = np.random.default_rng(1)
    for seed in range(5):
        noise = rng.normal(0, 1, 120)
        detected = detect_period(noise)
        assert detected is None or detected >= 2


def test_no_period_in_a_pure_trend():
    assert detect_period(np.linspace(0.0, 5.0, 120)) is None


def test_minimum_length_allows_the_harmonic_check():
    assert min_length_for_period(12) == 50
    assert min_length_for_period(4) == 18
    # One point short and the 2P harmonic cannot be examined, so no period.
    enough = min_length_for_period(PERIOD)
    assert detect_period(sawtooth(n=enough)) == PERIOD
    assert detect_period(sawtooth(n=enough - 1)) is None


def test_profile_recovers_the_planted_shape():
    profile = phase_profile(sawtooth(noise=0.02), PERIOD)
    assert int(np.nanargmin(profile.effect)) == 4
    assert int(np.nanargmax(profile.effect)) == 5
    assert profile.effect[4] < 0 < profile.effect[5]


def test_profile_is_centred_and_shrinks_toward_zero():
    raw = phase_profile(sawtooth(), PERIOD)
    assert np.nanmean(raw.effect) == pytest.approx(0.0, abs=1e-9)
    shrunk = phase_profile(sawtooth(), PERIOD, shrinkage=5.0)
    assert np.nanmax(np.abs(shrunk.effect)) < np.nanmax(np.abs(raw.effect))
    with pytest.raises(ValueError):
        phase_profile(sawtooth(), PERIOD, shrinkage=-1.0)


def test_deseasonalizing_removes_the_cycle():
    series = sawtooth(noise=0.05)
    deseasonalized = deseasonalize(series, phase_profile(series, PERIOD))
    assert deseasonalized.std(ddof=1) < series.std(ddof=1)
    band = white_noise_band(len(series))
    before = acf(np.diff(series), 2 * PERIOD)
    after = acf(np.diff(deseasonalized), 2 * PERIOD)
    # The seasonal harmonics are what must disappear.
    assert before[PERIOD - 1] > band
    assert abs(after[PERIOD - 1]) < abs(before[PERIOD - 1])


def test_centered_moving_average_is_symmetric_and_nan_padded():
    flat = np.full(40, 7.0)
    trend = centered_moving_average(flat, PERIOD)
    assert np.isnan(trend[: PERIOD // 2]).all()
    assert np.isnan(trend[-(PERIOD // 2) :]).all()
    assert trend[PERIOD // 2 : -(PERIOD // 2)] == pytest.approx(7.0)


def test_acf_rejects_nan_and_bad_lags():
    with pytest.raises(ValueError):
        acf(np.array([1.0, np.nan, 3.0]), 1)
    with pytest.raises(ValueError):
        acf(np.arange(5.0), 5)
