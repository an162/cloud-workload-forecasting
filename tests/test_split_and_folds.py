"""Split and fold mechanics on a 60-timestamp frame (PLAN 8)."""

import numpy as np
import pandas as pd
import pytest

from cwfc.evaluate import fold_plan
from cwfc.seasonal import min_length_for_period
from cwfc.targets import ACTIVE_VM_COUNT, AGGREGATE_CPU, chronological_split
from cwfc.validation import expanding_window_folds

N = 60


def targets(n=N):
    return pd.DataFrame({
        "timestamp": np.arange(n) * 300,
        ACTIVE_VM_COUNT: np.full(n, 7),
        AGGREGATE_CPU: np.linspace(1.0, 2.0, n),
    })


def test_split_sizes_are_derived_not_stated():
    train, holdout = chronological_split(targets(), holdout_size=24)
    assert len(holdout) == 24
    assert len(train) == N - 24          # 36, never written down anywhere
    assert len(train) + len(holdout) == N


def test_split_is_chronological_and_contiguous():
    train, holdout = chronological_split(targets(), holdout_size=24)
    assert train["timestamp"].max() < holdout["timestamp"].min()
    assert list(holdout["timestamp"]) == list(targets()["timestamp"][-24:])


def test_split_sorts_unsorted_input():
    shuffled = targets().sample(frac=1.0, random_state=0)
    train, holdout = chronological_split(shuffled, holdout_size=10)
    assert train["timestamp"].is_monotonic_increasing
    assert holdout["timestamp"].is_monotonic_increasing
    assert train["timestamp"].max() < holdout["timestamp"].min()


def test_split_rejects_impossible_holdout():
    with pytest.raises(ValueError):
        chronological_split(targets(), holdout_size=N)
    with pytest.raises(ValueError):
        chronological_split(targets(), holdout_size=0)


def test_folds_never_let_train_reach_test():
    folds = expanding_window_folds(N, horizon=12, min_train=24, step=3)
    assert folds
    for f in folds:
        train_idx = range(*f.train_slice.indices(N))
        test_idx = range(*f.test_slice.indices(N))
        assert max(train_idx) < min(test_idx)
        assert len(test_idx) == 12
        assert set(train_idx).isdisjoint(test_idx)


def test_folds_are_fully_scored_and_ordered():
    folds = expanding_window_folds(N, horizon=12, min_train=24, step=3)
    assert all(f.origin + f.horizon <= N for f in folds)       # no partial paths
    origins = [f.origin for f in folds]
    assert origins == sorted(origins)
    assert origins[0] == 24
    assert max(origins) + 12 <= N


def test_fold_plan_respects_the_detector_minimum():
    # With a seasonal period the first origin must be able to test the 2P
    # harmonic, which needs 4P + 2 observations.
    folds = fold_plan(N, horizon=12, period=4, step=3)
    assert folds[0].origin == max(2 * 12, min_length_for_period(4))
    folds_long = fold_plan(N, horizon=6, period=12, step=3)
    assert folds_long[0].origin == min_length_for_period(12) == 50


def test_fold_plan_without_seasonality_uses_two_horizons():
    folds = fold_plan(N, horizon=12, period=None, step=3)
    assert folds[0].origin == 24


def test_too_short_series_yields_no_folds():
    assert expanding_window_folds(20, horizon=12, min_train=24, step=3) == []
