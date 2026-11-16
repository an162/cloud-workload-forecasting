"""Extension target: core-weighted utilisation (PLAN section 1).

Training window only, and **reported only** - this target did not influence the
frozen model choice and was never scored on the holdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _report import Report  # noqa: E402

from cwfc import frozen  # noqa: E402
from cwfc.data import load_cpu_readings, load_vm_metadata  # noqa: E402
from cwfc.evaluate import BASELINES, model_summary, peak_metrics, rolling_origin_predictions  # noqa: E402
from cwfc.seasonal import detect_period, phase_profile  # noqa: E402
from cwfc.targets import (  # noqa: E402
    AGGREGATE_CPU,
    CORE_WEIGHTED_CPU,
    build_core_weighted_target,
    build_targets,
    chronological_split,
    parse_core_bucket,
)

DATA = "challenge_data"
H, STEP = frozen.HOLDOUT_SIZE, frozen.ORIGIN_STEP
model = frozen.SELECTED_MODEL_NAME

readings = load_cpu_readings(DATA)
meta = load_vm_metadata(DATA)
train, _ = chronological_split(build_targets(readings), holdout_size=H)
y = train[AGGREGATE_CPU].to_numpy(dtype=float)
P = detect_period(y)
preds = rolling_origin_predictions(
    y, horizon=H, step=STEP, models={**BASELINES, model: frozen.SELECTED_MODEL}
)

r = Report(
    "Core-weighted utilisation (extension target)",
    "`sum(avg_cpu * cores) / sum(cores)` per timestamp. Training window only, and "
    "**validation-only**: this series was never scored on the holdout and did not "
    "influence the frozen model choice.",
)

r.h2("Why weight by cores")
r.text("The required `aggregate_cpu_utilization` weights every VM equally, so a "
       "2-core VM at 90% counts as much as a 24-core VM at 10%. Capacity is bought "
       "in cores, so this is closer to what a planner provisions.")
r.table(pd.DataFrame([
    {"bucket": b, "cores": parse_core_bucket(b),
     "n_vms": int((meta["virtual_core_bucket"] == b).sum())}
    for b in sorted(meta["virtual_core_bucket"].unique(), key=parse_core_bucket)
]), index=False)
r.caveat("`\">24\"` is an open-ended bucket mapped to **24 cores, a stated lower "
         "bound**. It affects 10 of 10,000 VMs here, so the effect is negligible on "
         "this release but may not be on another.")

core_weighted = build_core_weighted_target(readings, meta)
cw_train, _ = chronological_split(core_weighted, holdout_size=H)
yw = cw_train[CORE_WEIGHTED_CPU].to_numpy(dtype=float)
models = {**BASELINES, model: frozen.SELECTED_MODEL}
cw_preds = rolling_origin_predictions(yw, horizon=H, step=STEP, models=models)
pw, pu = phase_profile(yw, detect_period(yw)), phase_profile(y, P)
r.table(pd.DataFrame({
    "statistic": ["mean", "sd", "detected period", "profile amplitude",
                  "correlation with unweighted", "frozen-model MAE", "MAE / series sd"],
    "core_weighted": [yw.mean(), yw.std(ddof=1), detect_period(yw), pw.amplitude,
                      np.corrcoef(yw, y)[0, 1],
                      model_summary(cw_preds).loc[model, "mae"],
                      model_summary(cw_preds).loc[model, "mae"] / yw.std(ddof=1)],
    "unweighted": [y.mean(), y.std(ddof=1), P, pu.amplitude, 1.0,
                   model_summary(preds).loc[model, "mae"],
                   model_summary(preds).loc[model, "mae"] / y.std(ddof=1)],
}), index=False)
r.table(model_summary(cw_preds))
r.kv([("total cores", f"{core_weighted['total_cores'].iloc[0]:,.0f} "
                      f"(constant: {core_weighted['total_cores'].nunique() == 1})"),
      ("'>24' bucket", "mapped to 24 cores, a stated lower bound")])
r.note("Capacity is bought in cores, not VMs. Relative to its own variability this "
       "series was easier to forecast in validation - MAE/sd 0.891 against 1.008 for "
       "the same model - and its hourly cycle is about 19% larger. The two series have "
       "different scales and variances, so **MAE/sd is the comparison to read**; the "
       "raw MAE figures are not comparable across scales. This is a **validation-only** "
       "result: the core-weighted series was never scored on the holdout and did not "
       "influence the frozen choice, so it is a lead worth pursuing rather than a "
       "switch to make.")


r.save("04_core_weighted.md")
