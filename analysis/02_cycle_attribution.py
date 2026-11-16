"""Who drives the hourly cycle and the bump, and can either be seen coming?

Training window only. Net and gross movement are always reported together.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _report import Report  # noqa: E402

from cwfc.attribution import (  # noqa: E402
    contribution_shares,
    cycle_jump_contributions,
    movement_summary,
    top_vm_shares,
    vm_timestamp_matrix,
    window_shift_contributions,
)
from cwfc.checks import infer_step  # noqa: E402
from cwfc.data import load_cpu_readings, load_vm_metadata  # noqa: E402
from cwfc.frozen import HOLDOUT_SIZE  # noqa: E402
from cwfc.monitor import detector_sweep, evaluate_detector, group_series, trailing_growth  # noqa: E402
from cwfc.seasonal import detect_period, phase_profile  # noqa: E402
from cwfc.targets import AGGREGATE_CPU, build_targets, chronological_split, valid_reading_mask  # noqa: E402

DATA = "challenge_data"
BUMP_START, BUMP_END = 53, 72          # read off the training series shape
THRESHOLDS = (2.0, 5.0, 10.0)          # CPU points of growth per cycle

readings = load_cpu_readings(DATA)
meta = load_vm_metadata(DATA)
train, _ = chronological_split(build_targets(readings), holdout_size=HOLDOUT_SIZE)
y = train[AGGREGATE_CPU].to_numpy(dtype=float)
train_ts = train["timestamp"].to_numpy()
N, P = len(y), detect_period(y)
step_seconds = infer_step(train_ts)

train_readings = readings[readings["timestamp"].isin(train_ts)]
train_valid = train_readings.loc[valid_reading_mask(train_readings)]
matrix = vm_timestamp_matrix(train_valid)
profile = phase_profile(y, P)
dip, rebound = int(np.nanargmin(profile.effect)), int(np.nanargmax(profile.effect))

jump = cycle_jump_contributions(matrix, P, dip, rebound)
bump = window_shift_contributions(matrix, slice(BUMP_START, BUMP_END + 1), slice(0, BUMP_START))

r = Report(
    "Cycle and bump attribution",
    f"Training window only ({N} timestamps). Per-VM contributions are signed and "
    "largely cancel, so **net and gross movement are reported side by side** "
    "throughout: net is what the aggregate shows, gross is the activity underneath it.",
)

r.h2("1. The two events")
rows = []
for label, contributions in (("hourly dip->rebound jump", jump), (f"bump (steps {BUMP_START}-{BUMP_END})", bump)):
    summary = movement_summary(contributions)
    n_vms = len(contributions)
    rows.append({
        "event": label,
        "net_points": summary["net"] / n_vms,
        "gross_points": summary["gross"] / n_vms,
        "cancellation": summary["cancellation_ratio"],
        "vms_up": summary["n_up"],
        "vms_down": summary["n_down"],
    })
r.table(pd.DataFrame(rows), index=False)
r.text(f"Pre-bump mean {y[:BUMP_START].mean():.4f}, bump mean "
       f"{y[BUMP_START:BUMP_END + 1].mean():.4f}, post-bump mean {y[BUMP_END + 1:].mean():.4f}.")
r.note("Both aggregate moves are small residuals of movement roughly 4x larger "
       "underneath. Roughly half the fleet moves against the aggregate in each event. "
       "A capacity signal taken from the aggregate alone therefore hides most of what "
       "is happening, and a per-VM alert tuned to aggregate-scale changes would fire "
       "constantly.")

r.h2("2. Who drives them: net vs gross")
for label, contributions in (("Hourly jump", jump), ("Bump", bump)):
    r.h3(f"{label}: top-contributing VMs")
    r.table(top_vm_shares(contributions), index=False)
    for column in ("vm_category", "virtual_core_bucket"):
        r.h3(f"{label}: by {column}")
        shares = contribution_shares(contributions, meta, column)
        r.table(shares[["n_vms", "net", "gross", "up", "down", "vm_share",
                        "share_of_net", "share_of_gross", "lift", "lift_gross"]])
    r.h3(f"{label}: by subscription")
    shares = contribution_shares(contributions, meta, "subscription_id")
    view = shares.head(10).copy()
    view.index = [f"{str(i)[:12]}.." for i in view.index]
    r.table(view[["n_vms", "net", "gross", "vm_share", "share_of_net",
                  "share_of_gross", "lift", "lift_gross"]])
    cumulative = pd.DataFrame([
        {"top_k_subscriptions": k,
         "share_of_net": float(shares["share_of_net"].head(k).sum()),
         "share_of_gross": float(
             shares.sort_values("share_of_gross", ascending=False)["share_of_gross"].head(k).sum())}
        for k in (1, 5, 10, 25)
    ])
    r.table(cumulative, index=False)

correlation = float(np.corrcoef(jump.to_numpy(), bump.reindex(jump.index).to_numpy())[0, 1])
r.kv([("correlation between a VM's jump and bump contribution", f"{correlation:+.3f}")])
r.text("Net shares exceed 100% and can go negative because groups move in opposite "
       "directions; that is why the gross column is always shown beside them. On the "
       "gross view the top 10 subscriptions account for ~18% of the hourly cycle's "
       "movement and ~28% of the bump's - real concentration, but three to four times "
       "less than the net view alone suggests.")
r.note("The hourly cycle and the bump are driven by **different populations** "
       "(correlation ~0.06). A segment rule written for one will not catch the other. "
       "The bump is the more concentrated event: a handful of subscriptions explain "
       "essentially all of its net move, which makes it a scheduling question for "
       "specific tenants rather than a fleet capacity question.")

r.h2("3. Can the bump be seen coming? A prospective test")
r.text("The retrospective version of this claim - 'the VMs that drove the bump were "
       "already ramping beforehand' - is true but worthless, because it selects the "
       "group using the outcome. Replaced here by a causal detector: at each step, "
       "using only data up to that step, flag any subscription whose trailing "
       f"{P}-step least-squares growth reaches a threshold. The driver labels are "
       "retrospective and used **for scoring only**; the detector never sees them.")

subscriptions = group_series(train_valid, meta, "subscription_id")
growth = trailing_growth(subscriptions, window=P)
eligible = int((~np.isnan(growth.to_numpy(dtype=float))).any(axis=1).sum())
flat = growth.to_numpy(dtype=float)
flat = flat[~np.isnan(flat)]
r.kv([
    ("subscriptions monitored", subscriptions.shape[1]),
    ("eligible steps", f"{eligible} of {N} "
                       f"({eligible * step_seconds / 3600:.2f} h monitored)"),
    ("growth distribution", f"mean {flat.mean():+.3f}, sd {flat.std():.3f}, "
                            f"p99 {np.percentile(flat, 99):.3f}, max {flat.max():.3f}"),
])

bump_shares = contribution_shares(bump, meta, "subscription_id")
labels = {
    "top-10 bump drivers": list(bump_shares.head(10).index),
    ">=1% of net bump": list(bump_shares[bump_shares["share_of_net"] >= 0.01].index),
}
for name, drivers in labels.items():
    r.h3(f"Label: {name} (n={len(drivers)})")
    r.table(detector_sweep(growth, drivers, BUMP_START, THRESHOLDS, step_seconds), index=False)

result = evaluate_detector(growth, labels["top-10 bump drivers"], BUMP_START, 5.0, step_seconds)
r.h3("Per-driver lead times at threshold 5.0")
r.table(pd.DataFrame([
    {"subscription": f"{str(k)[:12]}..", "lead_steps": v,
     "lead_minutes": v * step_seconds / 60}
    for k, v in result.lead_by_driver.items()
]), index=False)
r.kv([("drivers not flagged before the event",
       f"{result.n_drivers - result.n_detected_before_event} of {result.n_drivers}")])

r.text("Lead times look impressive - 95 to 210 minutes - but the maximum equals the "
       "first eligible step, meaning those subscriptions were already ramping before "
       "monitoring could begin. The precision is the problem.")
r.note("**This detector does not work as a fleet-wide alert.** Precision is 0.7-3.4% "
       "against the top-10 driver label, with 60-450 false-alarm episodes per hour "
       "across 2,324 subscriptions; at the loosest threshold 874 subscriptions (38%) "
       "are flagged before the event. Hundreds of tenants ramp as hard as the real "
       "drivers without moving the aggregate. Per subscription the rate is low "
       "(~0.03/hour), so this is usable as a *per-tenant* signal a team already "
       "watches - but the capacity policy must not depend on early detection.")
r.caveat("The driver label is retrospective. A flagged subscription that is not a "
         "top-10 driver is not necessarily a false signal - it grew, it just did not "
         "move the fleet aggregate. Precision against a narrow label understates the "
         "detector's value as a tenant-level signal and is reported as such.")

r.save("02_cycle_attribution.md")
