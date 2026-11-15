"""Leakage and submission-contract tests for the prediction pipeline (PLAN 8)."""

import numpy as np
import pandas as pd
import pytest

from cwfc import frozen
from cwfc.predict import forecast_targets, main
from cwfc.targets import ACTIVE_VM_COUNT, AGGREGATE_CPU

HOLDOUT = 12
N_TIMESTAMPS = 80
STEP = 300
RNG = np.random.default_rng(0)


def synthetic_readings(n_timestamps=N_TIMESTAMPS, n_vms=6, period=12):
    """A small panel with a real hourly-style cycle, so models have signal."""
    rows = []
    for i in range(n_timestamps):
        seasonal = 3.0 * np.sin(2 * np.pi * i / period)
        for v in range(n_vms):
            value = 20.0 + seasonal + v + RNG.normal(0, 0.3)
            rows.append((i * STEP, f"vm{v}", float(np.clip(value, 0, 100))))
    return pd.DataFrame(rows, columns=["timestamp", "vm_id", "avg_cpu"])


def corrupt_holdout(readings, holdout=HOLDOUT, value=999.0):
    """Overwrite every holdout CPU reading with garbage."""
    out = readings.copy()
    cutoff = np.sort(out["timestamp"].unique())[-holdout]
    out.loc[out["timestamp"] >= cutoff, "avg_cpu"] = value
    return out


def test_no_lookahead_holdout_values_cannot_change_predictions():
    readings = synthetic_readings()
    baseline = forecast_targets(readings, holdout_size=HOLDOUT)

    for garbage in (999.0, -5.0, 0.0, np.nan):
        corrupted = corrupt_holdout(readings, value=garbage)
        result = forecast_targets(corrupted, holdout_size=HOLDOUT)
        pd.testing.assert_frame_equal(baseline, result)


def test_no_lookahead_through_the_cli(tmp_path):
    readings = synthetic_readings()
    clean_dir, dirty_dir = tmp_path / "clean", tmp_path / "dirty"
    for path, frame in ((clean_dir, readings), (dirty_dir, corrupt_holdout(readings))):
        path.mkdir()
        frame.to_csv(path / "vm_cpu_readings.csv.gz", index=False, compression="gzip")

    clean_out, dirty_out = tmp_path / "clean.csv", tmp_path / "dirty.csv"
    for data_dir, out in ((clean_dir, clean_out), (dirty_dir, dirty_out)):
        assert main(["--data", str(data_dir), "--output", str(out),
                     "--holdout", str(HOLDOUT), "--quiet"]) == 0

    pd.testing.assert_frame_equal(pd.read_csv(clean_out), pd.read_csv(dirty_out))


def test_predictions_match_the_submission_schema(tmp_path):
    readings = synthetic_readings()
    out = tmp_path / "predictions.csv"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    readings.to_csv(data_dir / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
    main(["--data", str(data_dir), "--output", str(out), "--holdout", str(HOLDOUT), "--quiet"])

    frame = pd.read_csv(out)
    assert list(frame.columns) == list(frozen.SUBMISSION_COLUMNS)   # exact order
    assert len(frame) == HOLDOUT                                    # exact row count
    assert frame["timestamp"].is_unique
    assert frame["timestamp"].is_monotonic_increasing
    assert np.isfinite(frame.to_numpy(dtype=float)).all()
    assert (frame[ACTIVE_VM_COUNT] >= 0).all()
    assert frame[AGGREGATE_CPU].between(0.0, 100.0).all()

    expected = np.sort(readings["timestamp"].unique())[-HOLDOUT:]
    assert list(frame["timestamp"]) == list(expected)               # the right timestamps

    with open(out) as handle:
        assert handle.readline().strip() == ",".join(frozen.SUBMISSION_COLUMNS)


def test_predictions_are_bounded_even_when_the_model_is_not():
    """Output clipping is enforced regardless of what a model returns."""
    readings = synthetic_readings()
    readings.loc[readings["vm_id"] == "vm0", "avg_cpu"] = 99.9
    out = forecast_targets(readings, holdout_size=HOLDOUT)
    assert out[AGGREGATE_CPU].between(0.0, 100.0).all()
    assert (out[ACTIVE_VM_COUNT] >= 0).all()
    assert out[ACTIVE_VM_COUNT].dtype.kind in "iu"   # whole VMs, not fractions


def test_training_gap_is_refused_rather_than_filled():
    readings = synthetic_readings()
    grid = np.sort(readings["timestamp"].unique())
    readings = readings[readings["timestamp"] != grid[5]]   # gap inside training
    with pytest.raises(ValueError, match="gaps"):
        forecast_targets(readings, holdout_size=HOLDOUT)
