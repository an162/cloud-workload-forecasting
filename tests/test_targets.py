"""Target construction on a hand-made frame with every defect PLAN 8 lists."""

import numpy as np
import pandas as pd
import pytest

from cwfc.targets import (
    ACTIVE_VM_COUNT,
    AGGREGATE_CPU,
    build_targets,
    diagnose_targets,
    full_grid,
    out_of_range_mask,
    valid_reading_mask,
)

STEP = 300


def frame(rows):
    return pd.DataFrame(rows, columns=["timestamp", "vm_id", "avg_cpu"])


def clean():
    """Two VMs over three timestamps, no defects."""
    return frame([
        (0, "a", 10.0), (0, "b", 20.0),
        (300, "a", 30.0), (300, "b", 50.0),
        (600, "a", 1.0), (600, "b", 3.0),
    ])


def test_clean_targets():
    out = build_targets(clean())
    assert list(out["timestamp"]) == [0, 300, 600]
    assert list(out[ACTIVE_VM_COUNT]) == [2, 2, 2]
    assert out[AGGREGATE_CPU].tolist() == [15.0, 40.0, 2.0]


def test_identical_duplicate_counted_once_and_reported():
    data = pd.concat([clean(), frame([(0, "a", 10.0)])], ignore_index=True)
    out = build_targets(data)
    # The VM must not be double counted, and the mean must be unchanged.
    assert out.loc[0, ACTIVE_VM_COUNT] == 2
    assert out.loc[0, AGGREGATE_CPU] == 15.0
    report = diagnose_targets(data)
    assert report.n_duplicate_rows == 2
    assert report.n_identical_duplicate_groups == 1
    assert report.n_conflicting_duplicate_groups == 0


def test_conflicting_duplicate_is_averaged_then_counted_once():
    data = pd.concat([clean(), frame([(0, "a", 20.0)])], ignore_index=True)
    out = build_targets(data)
    assert out.loc[0, ACTIVE_VM_COUNT] == 2
    # VM "a" becomes (10+20)/2 = 15, so the timestamp mean is (15+20)/2.
    assert out.loc[0, AGGREGATE_CPU] == pytest.approx(17.5)
    report = diagnose_targets(data)
    assert report.n_conflicting_duplicate_groups == 1


def test_missing_reading_is_excluded_not_filled():
    data = pd.concat([clean(), frame([(300, "c", np.nan)])], ignore_index=True)
    out = build_targets(data)
    # The null-valued VM is not active and does not enter the mean.
    assert out.loc[1, ACTIVE_VM_COUNT] == 2
    assert out.loc[1, AGGREGATE_CPU] == 40.0
    assert diagnose_targets(data).n_invalid == 1


def test_gap_becomes_a_grid_slot_with_missing_targets():
    # Five timestamps so the modal step survives losing one of them.
    data = frame([(t, v, 1.0) for t in (0, 300, 600, 900, 1200) for v in ("a", "b")])
    data = data[data["timestamp"] != 600]  # punch a hole in the middle
    out = build_targets(data)
    # The grid still spans 0..1200 at step 300, so the gap stays visible
    # rather than silently shortening the series and shifting every lag.
    assert list(out["timestamp"]) == [0, 300, 600, 900, 1200]
    assert pd.isna(out.loc[2, ACTIVE_VM_COUNT])
    assert pd.isna(out.loc[2, AGGREGATE_CPU])
    report = diagnose_targets(data)
    assert report.n_grid_slots == 5
    assert report.n_slots_absent == 1
    assert report.step == STEP


def test_gap_is_not_inferable_from_two_timestamps():
    # With only 0 and 600 present, 600 *is* the modal step: a missing 300
    # cannot be distinguished from a series sampled every 600s. Pinned
    # deliberately so the limitation is documented rather than assumed away.
    data = clean()[lambda d: d["timestamp"] != 300]
    out = build_targets(data)
    assert list(out["timestamp"]) == [0, 600]
    assert diagnose_targets(data).n_slots_absent == 0


def test_timestamp_present_but_no_valid_reading_counts_zero_not_na():
    data = pd.concat([clean(), frame([(900, "a", np.nan)])], ignore_index=True)
    out = build_targets(data)
    slot = out.index[out["timestamp"] == 900][0]
    # Rows existed but none were usable: zero active VMs, undefined mean.
    # Zero and NA mean different things and must not collapse into each other.
    assert out.loc[slot, ACTIVE_VM_COUNT] == 0
    assert pd.isna(out.loc[slot, AGGREGATE_CPU])


def test_out_of_range_is_kept_and_flagged_not_dropped():
    data = pd.concat([clean(), frame([(0, "c", 150.0)])], ignore_index=True)
    out = build_targets(data)
    # 150 is implausible but real: it stays in both targets, and is reported.
    assert out.loc[0, ACTIVE_VM_COUNT] == 3
    assert out.loc[0, AGGREGATE_CPU] == pytest.approx((10 + 20 + 150) / 3)
    report = diagnose_targets(data)
    assert report.n_out_of_range == 1
    assert report.n_invalid == 0
    assert out_of_range_mask(data).sum() == 1
    assert valid_reading_mask(data).all()


def test_each_vm_counted_at_most_once_per_timestamp():
    data = frame([(0, "a", 1.0), (0, "a", 2.0), (0, "a", 3.0), (300, "a", 4.0)])
    out = build_targets(data)
    assert list(out[ACTIVE_VM_COUNT]) == [1, 1]


def test_full_grid_infers_step_and_fills_absent_slots():
    assert list(full_grid([0, 300, 900])) == [0, 300, 600, 900]
    assert list(full_grid([0, 5, 10, 20])) == [0, 5, 10, 15, 20]


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        build_targets(frame([]))
