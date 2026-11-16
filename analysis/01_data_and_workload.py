"""Data quality and workload analysis. CPU-series statistics: training only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _report import Report  # noqa: E402
from cwfc.checks import cadence_report
from cwfc.data import load_cpu_readings, load_vm_metadata
from cwfc.frozen import HOLDOUT_SIZE
from cwfc.seasonal import (
    acf,
    centered_moving_average,
    deseasonalize,
    detect_period,
    phase_profile,
    white_noise_band,
)
from cwfc.targets import (
    ACTIVE_VM_COUNT,
    AGGREGATE_CPU,
    build_targets,
    chronological_split,
    diagnose_targets,
    valid_reading_mask,
)

DATA = "challenge_data"
CPU_COLS = ["min_cpu", "avg_cpu", "max_cpu"]

readings = load_cpu_readings(DATA)
meta = load_vm_metadata(DATA)
targets = build_targets(readings)
train, holdout = chronological_split(targets, holdout_size=HOLDOUT_SIZE)
y = train[AGGREGATE_CPU].to_numpy(dtype=float)
train_ts = train["timestamp"].to_numpy()
N = len(y)
P = detect_period(y)

r = Report(
    "Data quality and workload analysis",
    "Every CPU-series statistic below uses the training window only "
    f"({N} of {len(targets)} timestamps). Structural checks over the whole file are "
    "labelled as such.",
)

# ---------------------------------------------------------------- grid
r.h2("1. Observation grid")
cadence = cadence_report(readings)
span = cadence.last_timestamp - cadence.first_timestamp
r.kv([
    ("distinct timestamps", cadence.n_timestamps),
    ("range", f"{cadence.first_timestamp} to {cadence.last_timestamp} (relative seconds)"),
    ("inferred step", f"{cadence.expected_step} s = {cadence.expected_step / 60:g} min (modal diff)"),
    ("all distinct diffs", cadence.step_counts),
    ("gaps", cadence.n_gaps),
    ("implied missing slots", cadence.n_missing_slots),
    ("span", f"{span} s = {span / 3600:.3f} h"),
    ("identity check", f"span/step + 1 = {span // cadence.expected_step + 1} "
                       f"vs {cadence.n_timestamps} observed"),
])
r.text("The cadence is inferred from the modal consecutive difference rather than "
       "assumed, and every distinct difference is reported beside it, so a "
       "half-irregular series would be visible instead of hidden behind the inference.")
r.note("A perfectly regular grid means any missingness must be a VM absent at an "
       "existing timestamp, or an unusable value in a row that is present - not a "
       "hole in the time axis. That narrows what monitoring has to watch for.")

# ---------------------------------------------------------------- panel
r.h2("2. Panel structure (structural, whole file)")
rows_per_vm = readings.groupby("vm_id").size()
distinct_per_vm = readings.groupby("vm_id")["timestamp"].nunique()
build = diagnose_targets(readings)
r.kv([
    ("rows", f"{build.n_rows:,}"),
    ("distinct VMs", f"{readings['vm_id'].nunique():,}"),
    ("rows per VM", f"min {rows_per_vm.min()}, max {rows_per_vm.max()}"),
    ("distinct timestamps per VM", f"min {distinct_per_vm.min()}, max {distinct_per_vm.max()}"),
    ("duplicate (vm_id, timestamp) rows", build.n_duplicate_rows),
    ("VMs covering the full grid", f"{int((distinct_per_vm == cadence.n_timestamps).sum()):,} "
                                   f"of {readings['vm_id'].nunique():,}"),
    ("absent grid slots", build.n_slots_absent),
])
r.text(f"With zero duplicates, `rows = VMs x timestamps` "
       f"({build.n_rows:,} = {readings['vm_id'].nunique():,} x {cadence.n_timestamps}) forces every VM to be "
       "present exactly once at every timestamp. `active_vm_count` is therefore "
       f"constant at {readings['vm_id'].nunique():,} - established from row counts, with no holdout "
       "target value read.")
r.note("This target is degenerate **by curation**: the release kept only VMs with "
       "complete traces, so zero churn is an artefact of dataset construction, not a "
       "property of Azure workloads. A real fleet has arrivals and departures, which "
       "would need to be modelled separately rather than forecast as one aggregate.")

# ---------------------------------------------------------------- validity
r.h2("3. Value validity (structural, whole file)")
rows = []
for col in CPU_COLS:
    values = pd.to_numeric(readings[col], errors="coerce")
    rows.append({
        "column": col,
        "null": int(values.isna().sum()),
        "non_finite": int((~np.isfinite(values)).sum()),
        "below_0": int((values < 0).sum()),
        "above_100": int((values > 100).sum()),
        "min": float(values.min()),
        "max": float(values.max()),
    })
r.table(pd.DataFrame(rows), index=False)
ordered = (readings["min_cpu"] <= readings["avg_cpu"]) & (readings["avg_cpu"] <= readings["max_cpu"])
r.kv([
    ("rows violating min <= avg <= max", f"{int((~ordered).sum())} of {len(readings):,}"),
    ("rows failing the validity predicate", int((~valid_reading_mask(readings)).sum())),
    ("valid-but-out-of-range rows", build.n_out_of_range),
])
r.text("The missing-data requirement is answered by a **documented proof of absence**: "
       "there is no missingness, no duplication and no range violation to treat. The "
       "cleaning paths still exist and are covered by tests, because a substituted "
       "dataset may not be this clean.")

# ---------------------------------------------------------------- series
r.h2("4. The aggregate CPU series (training window)")
diffs = np.diff(y)
r.kv([
    ("mean", f"{y.mean():.4f}%"),
    ("sd", f"{y.std(ddof=1):.4f}"),
    ("min / max", f"{y.min():.4f} / {y.max():.4f}"),
    ("total range", f"{y.max() - y.min():.4f} points"),
    ("sd of first differences", f"{diffs.std(ddof=1):.4f}"),
    ("mean absolute step change", f"{np.abs(diffs).mean():.4f}"),
])
r.text("Shape: flat near 9.4 for the first ~50 steps, a bump to ~9.9 around steps "
       "53-72, then a slow slide to ~9.15.")
r.note("The entire fleet aggregate moves inside a 0.8-point band around 9.4% "
       "utilisation. At that level the aggregate is nowhere near a capacity limit, so "
       "forecasting it precisely is not where operational risk lives - the value is in "
       "the shape and in the tail, not the level.")

# ---------------------------------------------------------------- seasonality
r.h2("5. The hourly cycle")
profile = phase_profile(y, P)
dip, rebound = int(np.nanargmin(profile.effect)), int(np.nanargmax(profile.effect))
deseasonalized = deseasonalize(y, profile)
band = white_noise_band(N)
raw_acf, des_acf = acf(y, 36), acf(deseasonalized, 36)
raw_diff, des_diff = acf(diffs, 36), acf(np.diff(deseasonalized), 36)

r.kv([
    ("detected period", f"{P} steps = {P * cadence.expected_step / 60:g} min"),
    ("detection rule", "largest differenced-ACF peak, accepted only if the ACF at both "
                       "that lag and twice that lag clear the white-noise band"),
    ("dip phase", f"{dip} (effect {profile.effect[dip]:+.4f})"),
    ("rebound phase", f"{rebound} (effect {profile.effect[rebound]:+.4f})"),
    ("amplitude", f"{profile.amplitude:.4f} points "
                  f"({profile.amplitude / np.abs(diffs).mean():.1f}x the mean step change)"),
])
r.table(pd.DataFrame({
    "phase": np.arange(P),
    "effect": profile.effect,
    "n_cycles": profile.counts,
}), index=False)

r.h3("Removing the profile reveals the real persistence")
r.table(pd.DataFrame({
    "statistic": ["ACF lag 1", "sd", "sd of first differences",
                  f"diff-ACF lag {P}", f"diff-ACF lag {2 * P}", f"diff-ACF lag {3 * P}",
                  "significant diff-ACF lags (of 36)"],
    "raw": [raw_acf[0], y.std(ddof=1), diffs.std(ddof=1),
            raw_diff[P - 1], raw_diff[2 * P - 1], raw_diff[3 * P - 1],
            int((np.abs(raw_diff) > band).sum())],
    "deseasonalized": [des_acf[0], deseasonalized.std(ddof=1), np.diff(deseasonalized).std(ddof=1),
                       des_diff[P - 1], des_diff[2 * P - 1], des_diff[3 * P - 1],
                       int((np.abs(des_diff) > band).sum())],
}), index=False)
r.text(f"White-noise band +/-{band:.4f}. Removing one {P}-step profile roughly halves "
       f"step-to-step volatility ({diffs.std(ddof=1):.4f} to "
       f"{np.diff(deseasonalized).std(ddof=1):.4f}) and eliminates all three seasonal "
       f"harmonics, while lag-1 autocorrelation rises from {raw_acf[0]:.3f} to "
       f"{des_acf[0]:.3f}.")
r.note("The raw lag-1 autocorrelation understates how smooth this workload actually "
       "is: the hourly sawtooth inflates apparent step-to-step noise. A team sizing "
       "alert thresholds off raw volatility would set them ~2x too wide. The cycle is "
       "predictable structure, not noise, and should be subtracted before anything "
       "else is measured.")

# ---------------------------------------------------------------- segments
r.h2("6. Workload by segment (training window)")
train_valid = readings[readings["timestamp"].isin(train_ts)]
train_valid = train_valid.loc[valid_reading_mask(train_valid)]
joined = train_valid.merge(
    meta[["vm_id", "vm_category", "virtual_core_bucket", "memory_gb_bucket"]],
    on="vm_id", how="left",
)
for column in ["vm_category", "virtual_core_bucket", "memory_gb_bucket"]:
    rows = []
    for name, group in joined.groupby(column):
        series = group.groupby("timestamp")["avg_cpu"].mean().reindex(train_ts)
        rows.append({
            column: name,
            "n_vms": group["vm_id"].nunique(),
            "level": float(series.mean()),
            "sd": float(series.std(ddof=1)),
            "corr_vs_aggregate": float(np.corrcoef(series.to_numpy(), y)[0, 1]),
        })
    r.table(pd.DataFrame(rows).sort_values("level", ascending=False), index=False)
r.text("Correlation with the aggregate tracks segment **size**, not segment behaviour: "
       "the largest core bucket correlates ~0.96 because it largely *is* the aggregate, "
       "while 10-VM buckets are an order of magnitude noisier and uncorrelated.")
r.note("Interactive VMs run hotter than delay-insensitive ones (11.5 vs 8.3). If "
       "headroom is ever tight, delay-insensitive capacity is the safer place to "
       "borrow from - that is a scheduling lever the aggregate forecast cannot see.")

# ---------------------------------------------------------------- concentration
r.h2("7. Demand concentration (training window)")
vm_mean = train_valid.groupby("vm_id")["avg_cpu"].mean()
quantiles = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
r.table(pd.DataFrame({
    "quantile": [f"p{int(q * 100)}" for q in quantiles],
    "per_VM_mean_cpu": [float(vm_mean.quantile(q)) for q in quantiles],
}), index=False)
total = float(vm_mean.sum())
descending = vm_mean.sort_values(ascending=False)
r.table(pd.DataFrame([
    {"top_share_of_vms": f"{share:.0%}",
     "n_vms": max(1, int(round(share * len(descending)))),
     "share_of_total_cpu": float(
         descending.iloc[: max(1, int(round(share * len(descending))))].sum() / total)}
    for share in (0.01, 0.10, 0.25, 0.50)
]), index=False)
r.kv([
    ("mean / median per-VM CPU", f"{vm_mean.mean():.3f} / {vm_mean.median():.3f} "
                                 f"(skew ratio {vm_mean.mean() / vm_mean.median():.2f})"),
    ("VMs averaging below 1%", f"{int((vm_mean < 1).sum()):,} ({(vm_mean < 1).mean():.1%})"),
    ("VMs averaging above 50%", f"{int((vm_mean > 50).sum()):,} ({(vm_mean > 50).mean():.1%})"),
])
r.note("The top 10% of VMs carry ~44% of all CPU and ~2% run above 50% utilisation, "
       "while ~6% sit below 1%. Capacity risk lives in that hot tail, not in the "
       "fleet mean - so the hot-VM share is the quantity worth alerting on, and the "
       "idle tail is where consolidation savings are.")

# ---------------------------------------------------------------- lifetimes
r.h2("8. Lifetimes are censored at both ends (structural)")
created, deleted = meta["timestamp_vm_created"], meta["timestamp_vm_deleted"]
max_deleted = int(deleted.max())
last_observed = int(readings["timestamp"].max())
r.kv([
    ("distinct creation timestamps", f"{created.nunique()} (value: {sorted(created.unique())})"),
    ("deletion range", f"{int(deleted.min())} to {max_deleted}"),
    ("deletions at exactly the maximum",
     f"{int((deleted == max_deleted).sum()):,} of {len(meta):,} "
     f"({(deleted == max_deleted).mean():.1%})"),
    ("maximum deletion timestamp in days", f"{max_deleted / 86400:.3f} d "
     f"(= 30 d minus {30 * 86400 - max_deleted} s)"),
    ("deletions after the last observed timestamp",
     f"{int((deleted > last_observed).sum()):,} of {len(meta):,}"),
    ("distinct deletion timestamps", deleted.nunique()),
])
r.text("Creation is a single value (0) for every VM, and 82% of deletions sit on one "
       "timestamp that corresponds to the upstream 30-day trace boundary.")
r.caveat("Both lifecycle fields are **censored at the trace edges**: creation at 0 means "
         "'already running when the trace started' (left-censored) and the modal deletion "
         "is the trace end (right-censored), not a real deletion. The sample is "
         "survivors only. Every conclusion here describes long-lived VMs observed "
         "mid-life, which need not behave like a fleet that includes short-lived ones.")

r.save("01_data_and_workload.md")
