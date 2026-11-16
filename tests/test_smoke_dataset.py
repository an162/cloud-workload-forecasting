"""End-to-end run on a tiny substituted dataset.

SUBMISSION.md warns that reviewers may "replace the public challenge data
with a tiny internal smoke dataset to verify that your pipeline is not
hard-coded to the supplied files". This builds such a dataset - roughly 200
VMs over 60 timestamps, with deleted rows and an injected duplicate - and
drives the documented prediction command over it.

Nothing here matches the real release: not the VM count, the timestamp count,
the cadence, the seasonal period, or the horizon.
"""

import numpy as np
import pandas as pd
import pytest

from cwfc import frozen
from cwfc.data import load_cpu_readings, load_vm_metadata
from cwfc.predict import capacity_band, main
from cwfc.targets import (
    ACTIVE_VM_COUNT,
    AGGREGATE_CPU,
    CORE_WEIGHTED_CPU,
    build_core_weighted_target,
    build_targets,
    diagnose_targets,
)

N_VMS = 200
N_TIMESTAMPS = 60
STEP = 120          # deliberately not 300
PERIOD = 6          # deliberately not 12
HOLDOUT = 24
N_DELETED = 7
CORE_BUCKETS = ("2", "4", "8", "24", ">24")


def build_smoke_dataset(tmp_path, seed=7):
    """Write a small, deliberately imperfect release-shaped dataset."""
    rng = np.random.default_rng(seed)
    timestamps = np.arange(N_TIMESTAMPS) * STEP
    vm_ids = [f"smoke-vm-{i:04d}" for i in range(N_VMS)]

    rows = []
    for index, timestamp in enumerate(timestamps):
        seasonal = 2.5 * np.sin(2 * np.pi * index / PERIOD)
        for offset, vm_id in enumerate(vm_ids):
            value = 18.0 + seasonal + (offset % 11) + rng.normal(0, 0.4)
            rows.append((int(timestamp), vm_id, float(np.clip(value, 0.0, 100.0))))
    readings = pd.DataFrame(rows, columns=["timestamp", "vm_id", "avg_cpu"])

    # Delete a handful of VM-timestamp rows, spread across the series, so the
    # panel is ragged and active_vm_count genuinely varies.
    victims = rng.choice(len(readings), size=N_DELETED, replace=False)
    readings = readings.drop(index=readings.index[victims]).reset_index(drop=True)

    # Inject one conflicting duplicate: same (vm_id, timestamp), different value.
    seed_row = readings.iloc[0]
    duplicate = pd.DataFrame(
        [(int(seed_row["timestamp"]), seed_row["vm_id"], float(seed_row["avg_cpu"]) + 4.0)],
        columns=["timestamp", "vm_id", "avg_cpu"],
    )
    readings = pd.concat([readings, duplicate], ignore_index=True)

    metadata = pd.DataFrame({
        "vm_id": vm_ids,
        "subscription_id": [f"sub-{i % 17}" for i in range(N_VMS)],
        "deployment_id": [f"dep-{i % 23}" for i in range(N_VMS)],
        "timestamp_vm_created": 0,
        "timestamp_vm_deleted": int(timestamps[-1]) + STEP,
        "max_cpu": 50.0,
        "avg_cpu": 20.0,
        "p95_max_cpu": 40.0,
        "vm_category": ["Interactive" if i % 3 else "Delay-insensitive" for i in range(N_VMS)],
        "virtual_core_bucket": [CORE_BUCKETS[i % len(CORE_BUCKETS)] for i in range(N_VMS)],
        "memory_gb_bucket": [str(2 ** (1 + i % 5)) for i in range(N_VMS)],
    })

    data_dir = tmp_path / "smoke_data"
    data_dir.mkdir()
    readings.to_csv(data_dir / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
    metadata.to_csv(data_dir / "vm_metadata.csv.gz", index=False, compression="gzip")
    return data_dir, readings, metadata


def test_smoke_dataset_has_the_defects_it_claims(tmp_path):
    data_dir, readings, _ = build_smoke_dataset(tmp_path)
    report = diagnose_targets(load_cpu_readings(data_dir))

    assert report.n_conflicting_duplicate_groups == 1
    assert report.n_duplicate_rows == 2
    assert report.n_grid_slots == N_TIMESTAMPS
    assert report.n_slots_absent == 0          # whole timestamps all survive
    assert report.step == STEP

    counts = build_targets(load_cpu_readings(data_dir))[ACTIVE_VM_COUNT]
    # Deleted rows must show up as a varying, below-full VM count.
    assert counts.max() == N_VMS
    assert counts.min() < N_VMS
    assert counts.nunique() > 1


def test_predict_runs_end_to_end_on_the_smoke_dataset(tmp_path):
    data_dir, readings, _ = build_smoke_dataset(tmp_path)
    output = tmp_path / "predictions.csv"

    assert main(["--data", str(data_dir), "--output", str(output),
                 "--holdout", str(HOLDOUT), "--quiet"]) == 0

    frame = pd.read_csv(output)
    assert list(frame.columns) == list(frozen.SUBMISSION_COLUMNS)
    assert len(frame) == HOLDOUT
    assert frame["timestamp"].is_unique and frame["timestamp"].is_monotonic_increasing
    assert np.isfinite(frame.to_numpy(dtype=float)).all()
    assert (frame[ACTIVE_VM_COUNT] >= 0).all()
    assert frame[AGGREGATE_CPU].between(0.0, 100.0).all()

    expected = np.sort(readings["timestamp"].unique())[-HOLDOUT:]
    assert list(frame["timestamp"]) == list(expected)
    # Scale is inherited from the data, not from the real release.
    assert frame[AGGREGATE_CPU].between(10.0, 35.0).all()


def test_smoke_dataset_core_weighted_target_builds(tmp_path):
    data_dir, _, metadata = build_smoke_dataset(tmp_path)
    core_weighted = build_core_weighted_target(
        load_cpu_readings(data_dir), load_vm_metadata(data_dir)
    )
    assert len(core_weighted) == N_TIMESTAMPS
    assert core_weighted[CORE_WEIGHTED_CPU].notna().all()
    assert core_weighted[CORE_WEIGHTED_CPU].between(0.0, 100.0).all()
    # ">24" is mapped to its stated lower bound of 24 cores.
    expected_cores = sum(float(b.lstrip(">")) for b in
                         (CORE_BUCKETS[i % len(CORE_BUCKETS)] for i in range(N_VMS)))
    assert core_weighted["total_cores"].max() == expected_cores


def test_band_refuses_rather_than_inventing_one_on_too_little_data(tmp_path):
    """A capacity band with no validation folds behind it must not ship."""
    data_dir, _, _ = build_smoke_dataset(tmp_path)
    readings = load_cpu_readings(data_dir)
    predictions = pd.DataFrame({
        "timestamp": np.sort(readings["timestamp"].unique())[-HOLDOUT:],
        AGGREGATE_CPU: np.full(HOLDOUT, 20.0),
    })
    with pytest.raises(ValueError, match="too short"):
        capacity_band(readings, predictions, holdout_size=HOLDOUT)
