from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cwfc.checks import infer_step

ACTIVE_VM_COUNT = "active_vm_count"
AGGREGATE_CPU = "aggregate_cpu_utilization"

CPU_LOWER = 0.0
CPU_UPPER = 100.0


def valid_reading_mask(frame: pd.DataFrame, value_col: str = "avg_cpu") -> pd.Series:
    """Rows whose CPU value is usable as an observation: non-null and finite.

    Range is deliberately NOT part of validity (PLAN 1.1): an out-of-range
    value is real data from the source and is kept and flagged rather than
    silently dropped or clipped. Use `out_of_range_mask` to report those.
    """
    values = pd.to_numeric(frame[value_col], errors="coerce")
    return values.notna() & np.isfinite(values)


def out_of_range_mask(
    frame: pd.DataFrame,
    value_col: str = "avg_cpu",
    lower: float = CPU_LOWER,
    upper: float = CPU_UPPER,
) -> pd.Series:
    """Valid-but-out-of-range rows, i.e. finite values outside [lower, upper]."""
    values = pd.to_numeric(frame[value_col], errors="coerce")
    return valid_reading_mask(frame, value_col) & ~values.between(lower, upper)


def full_grid(timestamps, step: int | None = None) -> pd.Index:
    """Complete arithmetic timestamp grid spanning the observed range.

    Built from min, max and the modal step, so a timestamp missing from the
    source appears as a grid slot with no data (NaN targets) instead of
    vanishing from the series and shortening it.
    """
    uniq = pd.Index(pd.unique(pd.Series(timestamps).to_numpy())).sort_values()
    if len(uniq) < 2:
        return uniq.astype("int64")
    s = infer_step(uniq) if step is None else int(step)
    if s <= 0:
        raise ValueError("step must be positive")
    return pd.Index(np.arange(int(uniq[0]), int(uniq[-1]) + s, s, dtype="int64"))


@dataclass(frozen=True)
class TargetBuildReport:
    """What the target build saw and did. Reported, never silently applied."""

    n_rows: int
    n_invalid: int
    n_out_of_range: int
    n_duplicate_rows: int
    n_identical_duplicate_groups: int
    n_conflicting_duplicate_groups: int
    n_grid_slots: int
    n_slots_present: int
    n_slots_absent: int
    n_slots_without_valid_data: int
    step: int


def _prepare(
    readings: pd.DataFrame,
    timestamp_col: str,
    vm_col: str,
    value_col: str,
    lower: float,
    upper: float,
) -> tuple[pd.DataFrame, pd.Index, TargetBuildReport]:
    """Shared core: validate, de-duplicate, and resolve the grid."""
    if readings.empty:
        raise ValueError("readings must not be empty")

    frame = readings.loc[:, [timestamp_col, vm_col, value_col]]
    n_rows = len(frame)

    valid = valid_reading_mask(frame, value_col)
    n_invalid = int((~valid).sum())
    n_out_of_range = int(out_of_range_mask(frame, value_col, lower, upper).sum())
    frame = frame.loc[valid]

    duplicated = frame.duplicated(subset=[vm_col, timestamp_col], keep=False)
    n_duplicate_rows = int(duplicated.sum())
    n_identical = n_conflicting = 0
    if n_duplicate_rows:
        spread = frame.loc[duplicated].groupby([vm_col, timestamp_col])[value_col].nunique()
        n_identical = int((spread == 1).sum())
        n_conflicting = int((spread > 1).sum())
        # Averaging collapses identical duplicates to the same value and
        # resolves conflicting ones, implementing both rules in one step.
        frame = frame.groupby([vm_col, timestamp_col], as_index=False)[value_col].mean()

    grid = full_grid(readings[timestamp_col])
    step = infer_step(grid) if len(grid) > 1 else 0
    present = grid.isin(pd.unique(readings[timestamp_col]))
    with_valid = grid.isin(pd.unique(frame[timestamp_col]))

    report = TargetBuildReport(
        n_rows=n_rows,
        n_invalid=n_invalid,
        n_out_of_range=n_out_of_range,
        n_duplicate_rows=n_duplicate_rows,
        n_identical_duplicate_groups=n_identical,
        n_conflicting_duplicate_groups=n_conflicting,
        n_grid_slots=len(grid),
        n_slots_present=int(present.sum()),
        n_slots_absent=int((~present).sum()),
        n_slots_without_valid_data=int((~with_valid).sum()),
        step=step,
    )
    return frame, grid, report


def build_targets(
    readings: pd.DataFrame,
    timestamp_col: str = "timestamp",
    vm_col: str = "vm_id",
    value_col: str = "avg_cpu",
    lower: float = CPU_LOWER,
    upper: float = CPU_UPPER,
) -> pd.DataFrame:
    """Build the two required aggregate target series from VM-level readings.

    One row per grid timestamp, chronological:

    - `active_vm_count`           distinct VMs with a valid reading at t
    - `aggregate_cpu_utilization` mean of valid VM-level `avg_cpu` at t

    Missing data is never filled. A grid slot with rows but no valid reading
    gets a count of 0 (VMs were there, none reported usably) and a NaN mean.
    A slot absent from the source entirely gets NA for both, so a gap stays
    visibly a gap instead of becoming a zero.
    """
    frame, grid, _ = _prepare(readings, timestamp_col, vm_col, value_col, lower, upper)

    counts = frame.groupby(timestamp_col)[vm_col].nunique().reindex(grid).to_numpy(dtype="float64")
    means = frame.groupby(timestamp_col)[value_col].mean().reindex(grid).to_numpy(dtype="float64")

    present = np.asarray(grid.isin(pd.unique(readings[timestamp_col])))
    counts = np.where(np.isnan(counts) & present, 0.0, counts)

    return pd.DataFrame(
        {
            timestamp_col: grid.to_numpy(dtype="int64"),
            ACTIVE_VM_COUNT: pd.array(counts, dtype="Float64").astype("Int64"),
            AGGREGATE_CPU: means,
        }
    )


def diagnose_targets(
    readings: pd.DataFrame,
    timestamp_col: str = "timestamp",
    vm_col: str = "vm_id",
    value_col: str = "avg_cpu",
    lower: float = CPU_LOWER,
    upper: float = CPU_UPPER,
) -> TargetBuildReport:
    """Return what `build_targets` would report about this input."""
    _, _, report = _prepare(readings, timestamp_col, vm_col, value_col, lower, upper)
    return report


def chronological_split(
    targets: pd.DataFrame,
    holdout_size: int,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a target frame into (train, holdout) by trailing timestamp count.

    The holdout is the last `holdout_size` rows; training is everything before.
    The training length is derived as `len(targets) - holdout_size`, never
    stated, so a substituted smaller dataset splits correctly unchanged.
    """
    if holdout_size <= 0:
        raise ValueError("holdout_size must be positive")
    if holdout_size >= len(targets):
        raise ValueError(
            f"holdout_size={holdout_size} leaves no training rows for {len(targets)} timestamps"
        )

    ordered = targets.sort_values(timestamp_col).reset_index(drop=True)
    cut = len(ordered) - holdout_size
    return ordered.iloc[:cut].reset_index(drop=True), ordered.iloc[cut:].reset_index(drop=True)


def timestamp_grid(frame: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.Index:
    """Sorted distinct timestamps actually present in `frame`."""
    return pd.Index(pd.unique(frame[timestamp_col])).sort_values()


CORE_WEIGHTED_CPU = "core_weighted_cpu_utilization"
CORE_BUCKET_COLUMN = "virtual_core_bucket"


def parse_core_bucket(value) -> float:
    """Core count implied by a `virtual_core_bucket` label.

    Buckets are labels, not measurements: ">24" is an open-ended bucket, and
    this maps it to 24 - a deliberate **lower bound**, stated rather than
    guessed. Any VM in that bucket has at least 24 cores, so the resulting
    weight understates those VMs. On this release only 10 of 10,000 VMs fall
    there, so the effect is negligible; on another dataset it may not be.
    """
    text = str(value).strip().lstrip(">").lstrip("<").strip()
    if not text:
        raise ValueError(f"cannot parse core bucket {value!r}")
    return float(text)


def build_core_weighted_target(
    readings: pd.DataFrame,
    meta: pd.DataFrame,
    timestamp_col: str = "timestamp",
    vm_col: str = "vm_id",
    value_col: str = "avg_cpu",
    bucket_col: str = CORE_BUCKET_COLUMN,
) -> pd.DataFrame:
    """Core-weighted fleet utilisation (PLAN section 1, extension target).

    `sum(avg_cpu * cores) / sum(cores)` at each timestamp, over valid readings.

    The required `aggregate_cpu_utilization` weights every VM equally, so a
    2-core VM at 90% counts as much as a 24-core VM at 10%. Capacity is
    provisioned in cores, not in VMs, so this is the quantity a planner
    actually cares about. Reported as an extension; it does not replace the
    required target.

    VMs missing from `meta` are dropped and the loss is detectable by
    comparing `n_vms` against the unweighted count.
    """
    frame, grid, _ = _prepare(
        readings, timestamp_col, vm_col, value_col, CPU_LOWER, CPU_UPPER
    )
    cores = (
        meta[[vm_col, bucket_col]]
        .assign(**{"_cores": lambda d: d[bucket_col].map(parse_core_bucket)})
        .set_index(vm_col)["_cores"]
    )
    joined = frame.join(cores, on=vm_col, how="inner")
    joined["_weighted"] = joined[value_col] * joined["_cores"]

    grouped = joined.groupby(timestamp_col)
    weighted_sum = grouped["_weighted"].sum().reindex(grid)
    core_sum = grouped["_cores"].sum().reindex(grid)
    n_vms = grouped[vm_col].nunique().reindex(grid)

    with np.errstate(invalid="ignore", divide="ignore"):
        values = (weighted_sum / core_sum).to_numpy(dtype="float64")

    return pd.DataFrame(
        {
            timestamp_col: grid.to_numpy(dtype="int64"),
            CORE_WEIGHTED_CPU: values,
            "total_cores": core_sum.to_numpy(dtype="float64"),
            "n_vms": n_vms.to_numpy(dtype="float64"),
        }
    )
