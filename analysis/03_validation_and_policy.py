"""Validation, model selection evidence, and the capacity policy.

Training window only. This script reproduces the evidence the frozen choice
was made on; it does not read holdout target values.
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
from cwfc.evaluate import (  # noqa: E402
    BASELINES,
    failure_analysis,
    fold_metrics,
    fold_plan,
    horizon_metrics,
    model_summary,
    peak_metrics,
    predictability_curve,
    rolling_origin_predictions,
)
from cwfc.models import CANDIDATES, ar1_phi  # noqa: E402
from cwfc.policy import calibrate_nominal_q, final_band_increments, policy_table, replay_policy, select_nominal_q  # noqa: E402
from cwfc.seasonal import deseasonalize, detect_period, min_length_for_period, phase_profile  # noqa: E402
from cwfc.targets import (  # noqa: E402
    AGGREGATE_CPU,
    CORE_WEIGHTED_CPU,
    build_core_weighted_target,
    build_targets,
    chronological_split,
)

DATA = "challenge_data"
H, STEP = frozen.HOLDOUT_SIZE, frozen.ORIGIN_STEP
ALL_MODELS = {**BASELINES, **CANDIDATES}

readings = load_cpu_readings(DATA)
meta = load_vm_metadata(DATA)
train, _ = chronological_split(build_targets(readings), holdout_size=H)
y = train[AGGREGATE_CPU].to_numpy(dtype=float)
N, P = len(y), detect_period(y)

r = Report(
    "Validation, selection and capacity policy",
    "Training window only. This is the evidence base the frozen model was selected "
    "on, reproduced; no holdout target value is read here.",
)
r.text("""**Contents**

1. [Rolling-origin design](#1-rolling-origin-design)
2. [Model comparison across folds](#2-model-comparison-across-folds) — and the
   paired tie-band that replaced the fold-sd one
3. [Per-fold detail](#3-per-fold-detail)
4. [Horizon behaviour and the predictability horizon](#4-horizon-behaviour-and-the-predictability-horizon)
5. [Peak behaviour](#5-peak-behaviour)
6. [Failure analysis and the limits of the evidence](#6-failure-analysis-and-the-limits-of-the-evidence)
   — including why the reversion evidence is weak
7. [Capacity band and policy](#7-capacity-band-and-policy)
8. Extension: core-weighted utilisation → **separate file**,
   [`04_core_weighted.md`](04_core_weighted.md)
""")

r.h2("1. Rolling-origin design")
folds = fold_plan(N, horizon=H, period=P, step=STEP)
r.kv([
    ("scheme", "expanding window inside the training window; fit on [0, o), forecast [o, o+H)"),
    ("N / H / P", f"{N} / {H} / {P}"),
    ("min_train", f"max(2H, min_length_for_period(P)) = max({2 * H}, {min_length_for_period(P)}) "
                  f"= {max(2 * H, min_length_for_period(P))}"),
    ("origin step", STEP),
    ("folds", f"{len(folds)}, origins {folds[0].origin}, {folds[1].origin}, ..., {folds[-1].origin}"),
    ("last origin", f"{folds[-1].origin} (= N - H = {N - H})"),
])
r.text("The first origin is the shortest window on which the period detector can test "
       "the 2P harmonic, so every fold answers the seasonality question for the same "
       "reason rather than one fold silently running non-seasonal. An origin step "
       "above 1 spreads origins across phases of the hourly cycle.")
r.caveat("Folds overlap - expanding origins share every earlier observation - so "
         "per-fold errors are correlated. Spread is reported as an indicator only and "
         "no significance test is implied anywhere in this report.")

preds = rolling_origin_predictions(y, horizon=H, step=STEP, models=ALL_MODELS)
per_fold_period = preds.groupby("fold")["period"].first()
r.kv([("per-fold detected period", per_fold_period.value_counts(dropna=False).to_dict())])

r.h2("2. Model comparison across folds")
summary = model_summary(preds)
r.table(summary)
r.text("`train_mean` beating every recency-anchored baseline is the result that shaped "
       "the candidate set: over a 24-step path the level estimate mattered more than "
       "the seasonal shape, so the candidates vary the level estimator while holding "
       "the phase profile fixed.")

r.h3("Paired per-fold differences against the leader")
leader = summary.index[0]
mae = fold_metrics(preds).pivot(index="fold", columns="model", values="mae")
rows = []
for model in mae.columns:
    difference = mae[model] - mae[leader]
    standard_error = difference.std(ddof=1) / np.sqrt(len(difference))
    rows.append({
        "model": model,
        "mean_diff": difference.mean(),
        "se_diff": standard_error,
        "mean_over_se": difference.mean() / standard_error if standard_error > 0 else np.nan,
        "folds_worse": int((difference > 0).sum()),
        "folds_better": int((difference < 0).sum()),
    })
r.table(pd.DataFrame(rows).sort_values("mean_diff"), index=False)
r.text(f"The tie-band is one standard error of the **paired** difference, not the "
       f"spread of the leader's own MAE across folds. Pairing cancels the shared fold "
       f"difficulty that made the unpaired band useless here: it would have spanned "
       f"{summary.loc[leader, 'mae']:.4f} to "
       f"{summary.loc[leader, 'mae'] + summary.loc[leader, 'mae_sd']:.4f} and admitted "
       f"five of nine candidates, selecting `train_mean`. See DECISIONS entry 19.")

r.h2("3. Per-fold detail")
per_fold = mae.copy()
per_fold.insert(0, "origin", preds.groupby("fold")["origin"].first())
r.table(per_fold)
r.table(pd.DataFrame({"wins": mae.idxmin(axis=1).value_counts()}))

r.h2("4. Horizon behaviour and the predictability horizon")
r.table(horizon_metrics(preds).pivot(index="step", columns="model", values="mae"))
curve = predictability_curve(preds).pivot(index="step", columns="model", values="skill_vs_train_mean")
r.table(curve, floatfmt=3)
skill = curve["persistence"]
negative = skill[skill <= 0]
first_negative = int(negative.index[0]) if len(negative) else None
step_minutes = 5
r.kv([
    ("persistence skill vs train_mean, steps 1-8",
     ", ".join(f"{v:+.3f}" for v in skill.to_numpy()[:8])),
    ("first step with non-positive skill",
     f"{first_negative} ({first_negative * step_minutes} minutes)" if first_negative else "none"),
    ("skill at the final step", f"{skill.iloc[-1]:+.3f}"),
])
r.note(f"Recent history stops paying after roughly {first_negative - 1} steps, i.e. "
       f"**about {(first_negative - 1) * step_minutes} minutes**. Beyond that the "
       "long-run level predicts better than anything the last few observations "
       "contain. A team should treat short-horizon autoscaling and two-hour capacity "
       "planning as different problems with different inputs.")
r.caveat("The crossing is noisy at this sample size - skill dips negative at step "
         f"{first_negative}, recovers briefly, then stays negative. The boundary is "
         "approximate, not a sharp threshold.")

r.h2("5. Peak behaviour")
r.table(peak_metrics(preds))
r.note("Every model that wins on average under-forecasts peaks (bias -0.31 to -0.34). "
       "That tension is handled by the band in section 7 rather than by biasing a "
       "model, so the safety margin stays explicit and costed in one place.")

r.h2("6. Failure analysis and the limits of the evidence")
rows = [failure_analysis(preds, model) for model in summary.index]
r.table(pd.DataFrame(rows)[[
    "model", "worst_fold", "worst_fold_origin", "worst_fold_mae", "best_fold",
    "best_fold_mae", "worst_step_position", "worst_step_h", "worst_step_abs_error",
]], index=False)

phis = []
for fold in folds:
    history = y[fold.train_slice]
    phis.append(ar1_phi(deseasonalize(history, phase_profile(history, P, shrinkage=frozen.PROFILE_SHRINKAGE))))
r.h3("Why the reversion evidence is weak")
r.table(pd.DataFrame({
    "fold": [f.index for f in folds],
    "origin": [f.origin for f in folds],
    "fitted_phi": phis,
}), index=False)
r.text(f"The fitted reversion rate jumps from ~{min(phis):.2f} in the earliest folds to "
       f"~{max(phis):.2f} in every later fold, and the jump coincides with the steps "
       f"53-72 bump entering the training window. At phi~{max(phis):.2f}, "
       f"phi^{H} = {max(phis) ** H:.4f}, so an AR(1) term is fully decayed before the "
       "horizon ends. This is why the AR(1) variant was dropped despite a nominally "
       "lower mean MAE (DECISIONS entries 18, 20).")
r.caveat("Thirteen overlapping folds inside a single 11-hour window contain **one** "
         "bump and **one** decline. The validation cannot distinguish 'mean-reverting' "
         "from 'slowly declining', which is the central limitation of everything in "
         "this section.")

r.h2("7. Capacity band and policy")
model = frozen.SELECTED_MODEL_NAME
selected_preds = preds[preds["model"] == model]
r.kv([
    ("policy claim", f"target breach rate {frozen.TARGET_BREACH_RATE:.0%}, i.e. running "
                     f"short costs {frozen.COST_RATIO:.0f}x what idling costs"),
    ("band form", "forecast + q-quantile of the under-forecast distribution, per horizon bucket"),
    ("width constraint", "non-decreasing across horizon buckets"),
])
raw = final_band_increments(preds, model, 0.90, monotone=False)
monotone = final_band_increments(preds, model, 0.90, monotone=True)
r.table(pd.DataFrame({
    "bucket": list(raw), "raw_q90": list(raw.values()),
    "monotone_q90": [monotone[b] for b in raw],
}), index=False)
r.text("The raw quantiles fall with horizon, which would claim a two-hour forecast "
       "needs less headroom than a thirty-minute one. That is the selected model's "
       "growing over-forecast during the post-bump slide, not uncertainty shrinking, "
       "so the widths are constrained non-decreasing.")

r.h3("Calibrating the nominal quantile")
sweep = calibrate_nominal_q(preds, model, frozen.TARGET_BREACH_RATE)
r.table(sweep[sweep["nominal_q"].isin([0.80, 0.85, 0.87, 0.875, 0.88, 0.90, 0.95])], index=False)
q = select_nominal_q(preds, model, frozen.TARGET_BREACH_RATE)
r.kv([("shipped nominal q", f"{q:.3f}"),
      ("shipped increments", {k: round(v, 4) for k, v in final_band_increments(preds, model, q).items()})])
r.text("A band built at nominal q=0.90 breached 12.5% of the time out of fold, not "
       "10%, so the nominal quantile is calibrated against the replayed breach rate "
       "rather than assumed to deliver it. The monotone constraint pushes the other "
       "way, which is why the shipped q came out below 0.90.")

r.h3("Replayed cost of the policy")
r.table(replay_policy(preds, model, q), index=False)
r.table(policy_table(preds, model, quantiles=(0.80, 0.90, 0.95)), index=False)
series_range = y.max() - y.min()
r.kv([
    ("band width at the shipped q", f"{np.mean(list(final_band_increments(preds, model, q).values())):.4f} points"),
    ("as a share of mean utilisation", f"{np.mean(list(final_band_increments(preds, model, q).values())) / y.mean():.2%}"),
    ("as a share of the series' total training range", f"{np.mean(list(final_band_increments(preds, model, q).values())) / series_range:.2f}x"),
])
r.note("At ~9.4% mean utilisation a 0.37-point band is operationally free, so the "
       "recommended policy is: provision to the band, not the point forecast, for "
       "two-hour planning. But the band is a **forecast-error** margin, not a capacity "
       "margin - it is smaller than the fleet's own hourly swing. The real levers are "
       "the hot-tail share (~2% of VMs above 50%) and the hourly rebound, neither of "
       "which the aggregate band addresses.")
r.caveat("Leave-one-fold-out removes a fold's own residuals but not the overlap "
         "between folds, so these breach rates remain optimistic and carry no "
         "confidence interval. Per-bucket rates are also uneven (~15% / 14% / 5%): one "
         "quantile plus a monotone constraint cannot equalise three buckets.")

r.text("The core-weighted extension target is reported separately in "
       "`04_core_weighted.md`.")

r.save("03_validation_and_policy.md")
