# Cloud Workload Forecasting — Final Report

**Dataset:** Azure Public Dataset V2, `data-v1` release — 10,000 VMs × 134
five-minute observations (11.08 h), 1,340,000 VM-level CPU readings.
**Task:** forecast the final 24 timestamps (2 h) of `active_vm_count` and
`aggregate_cpu_utilization`.
**Protocol:** model, band and policy frozen and git-tagged before the holdout
was scored; scored once.

---

## 1. Executive summary

The selected model **lost to a simple baseline on the holdout, and the loss was
predicted in writing beforehand.** That is the central result of this
submission, and it is a methodological finding rather than an accident.

| model | holdout MAE | bias | MASE | skill vs persistence |
|---|---:|---:|---:|---:|
| `last_cycle_mean` | **0.1230** | −0.053 | 1.016 | **+0.162** |
| `seasonal_naive` | 0.1400 | −0.053 | 1.156 | +0.046 |
| `persistence` | 0.1468 | −0.104 | 1.212 | 0.000 |
| **`profile_train_mean`** (selected) | 0.1609 | **+0.145** | 1.328 | **−0.096** |
| `train_mean` | 0.1746 | +0.145 | 1.442 | −0.189 |

Four findings carry the report:

1. **This workload is extremely stable and the aggregate is not where risk
   lives.** Fleet utilisation moves inside a 0.80-point band around 9.41%.
   Forecast errors of ~0.15 points are ~1.6% WAPE — accurate in relative terms
   and operationally irrelevant in absolute ones.
2. **There is a real 60-minute cycle, and removing it halves apparent
   volatility.** Subtracting a 12-step phase profile drops step-to-step sd from
   0.158 to 0.082 and lifts lag-1 autocorrelation from 0.662 to 0.879.
3. **Predictability expires after ~40 minutes.** Beyond ~8 steps, the long-run
   mean beats anything recent history contains. Short-horizon autoscaling and
   two-hour capacity planning are different problems.
4. **Thirteen overlapping folds inside one 11-hour window could not
   distinguish "mean-reverting" from "slowly declining."** That single
   ambiguity caused the model loss, and it is a limit of the *data*, not of the
   validation design.

**Recommendation:** use `last_cycle_mean` plus the hourly phase profile for
two-hour aggregate planning, provision to the band rather than the point
forecast, and monitor the hot tail (~2% of VMs above 50%) rather than the fleet
mean. Forecast **core-weighted** utilisation, not the unweighted mean.

---

## 2. Protocol and leakage controls

Because the holdout is public, the result depends on protocol discipline
rather than a hidden scoring service. Controls, all verifiable in the repo:

- **Nothing hard-coded.** `110`, `134` and `12` appear nowhere in code. The
  split derives the training size as `len(targets) − holdout_size`; the
  seasonal period is detected; the cadence is inferred from the modal diff.
  `tests/test_smoke_dataset.py` runs the documented command over a 200-VM ×
  60-timestamp dataset at 120 s cadence with period 6.
- **Per-fold fitting.** Period, phase profile, MASE scale, peak threshold and
  band quantiles are each fitted on the individual fold's training slice.
- **Separate entry points.** `cwfc.predict` reads only the holdout's
  `timestamp` column; `cwfc.score_holdout` is the single place holdout targets
  are read.
- **Tested, not asserted.** Overwriting every holdout reading with 999 / −1 /
  0 / NaN leaves predictions bit-identical (max abs diff 0.00e+00).
- **Frozen then tagged.** `src/cwfc/frozen.py`, git tag
  `frozen-before-holdout`, committed before scoring.
- **Metadata lifetime statistics** (`avg_cpu`, `max_cpu`, `p95_max_cpu`,
  `timestamp_vm_deleted`) are descriptive only, never features.

One disclosure: an early exploratory pass printed `active_vm_count.unique()`
across all 134 timestamps, which observes that target's holdout rows. It is
logged as a deliberate structural check (DECISIONS entry 7). The constant is
derivable from row counts alone — 1,340,000 = 10,000 × 134 with zero duplicates
— and the training window alone shows the same value, so no decision depended
on it. `aggregate_cpu_utilization` holdout values were unread until the gate.

---

## 3. Data understanding and preprocessing

**The data is pristine, and that is a finding to state rather than a step to
skip.** Across 1,340,000 rows: 0 nulls, 0 non-finite values, 0 duplicate
`(vm_id, timestamp)` pairs, 0 values outside [0,100], and `min ≤ avg ≤ max`
holds everywhere. The grid is perfectly regular — 134 timestamps, every
consecutive difference exactly 300 s, and `span/step + 1 = 134` exactly.

The missing-data requirement is therefore answered by a **documented proof of
absence**. The cleaning paths still exist and are tested, because a substituted
dataset may not be this clean: identical duplicates collapse, conflicting
duplicates average, an absent grid slot yields NA for both targets, and a
timestamp present with no usable reading yields count 0 and a NaN mean. Zero
and "no observation" are kept distinct and nothing is ever filled.

**Target definitions.** One validity predicate — `avg_cpu` non-null and finite
— serves both targets, so "VM is active at *t*" and "VM contributes to the mean
at *t*" cannot diverge. Out-of-range values are **kept and flagged**, not
dropped: an implausible reading is still real source data, and excluding it
would silently change the VM count.

**`active_vm_count` is degenerate by curation.** It is constant at 10,000. The
release's selection rule kept only VMs with complete 134-observation traces, so
zero churn is an artefact of dataset construction. All five models score MAE
0.0; MASE is NaN because the naive scale is zero. A real fleet has arrivals and
departures that would need separate modelling.

**Lifetimes are censored at both ends.** `timestamp_vm_created` is 0 for every
VM (left-censored: already running when the trace began). 8,213 of 10,000
deletions (82.1%) sit on one timestamp, 2,591,400 s — the upstream 30-day trace
boundary, i.e. right-censored, not real deletions. **The sample is survivors
only**, and every conclusion below describes long-lived VMs observed mid-life.

---

## 4. Workload analysis

### 4.1 The aggregate is remarkably flat

| mean | sd | min | max | range | sd(Δ) |
|---:|---:|---:|---:|---:|---:|
| 9.4141% | 0.1936 | 9.1116 | 9.9122 | **0.8005** | 0.1580 |

Shape: flat near 9.4 for ~50 steps, a bump to ~9.9 across steps 53–72, then a
slide to ~9.15.

### 4.2 A real 60-minute cycle

Detected period 12 steps = 60 min, accepted only because the differenced-series
ACF clears the white-noise band at both lag 12 (+0.608) and lag 24 (+0.576).
The profile has a sharp dip at phase 4 (−0.207) and rebound at phase 5
(+0.135) — amplitude 0.342 points, **2.8× the mean absolute step change**.

Removing it is the single most informative transformation in the analysis:

| | raw | deseasonalized |
|---|---:|---:|
| ACF lag 1 | 0.662 | **0.879** |
| sd of first differences | 0.1580 | **0.0816** |
| diff-ACF at lags 12 / 24 / 36 | +0.608 / +0.576 / +0.407 | −0.104 / +0.053 / −0.204 |
| significant diff-ACF lags (of 36) | 17 | **3** |

The raw lag-1 autocorrelation *understates* how smooth this workload is: the
hourly sawtooth inflates apparent step-to-step noise. A team sizing alert
thresholds off raw volatility would set them roughly twice too wide.

### 4.3 Demand is concentrated and right-skewed

Per-VM mean CPU: median 5.50%, mean 9.41% (skew ratio 1.71), p99 79.6%. The
top 1% of VMs carry 9.6% of all CPU, the top 10% carry 43.5%, the top 25% carry
64.5%. 5.8% of VMs average below 1%; **2.2% average above 50%**.

By segment: Interactive VMs run hotter (11.51) than Delay-insensitive (8.26).
Correlation with the aggregate tracks segment *size*, not behaviour — the
7,534-VM core bucket correlates 0.962 because it largely *is* the aggregate.

### 4.4 Both aggregate moves are small residuals of large offsetting activity

| event | net | gross | cancellation | VMs up / down |
|---|---:|---:|---:|---:|
| hourly dip→rebound | +0.349 | 1.383 | **4.0×** | 5,815 / 4,185 |
| bump (steps 53–72) | +0.349 | 1.468 | **4.2×** | 4,727 / 5,273 |

Roughly half the fleet moves *against* the aggregate in each event. Net and
gross shares are reported together throughout, because net-only shares exceed
100% (top 25 subscriptions = 142% of the net bump) and overstate concentration
3–4×: those same subscriptions are 45% of *gross* movement.

The two events have **different drivers** — per-VM contributions to the hourly
jump and to the bump correlate +0.056. The bump is far more concentrated (top
10 subscriptions = 98.3% of net, lifts 23–37×) than the cycle (38.3%).

### 4.5 The bump could not have been caught prospectively

A retrospective "the bump's drivers were already ramping" is true but
worthless — it selects the group using the outcome. The honest version is a
causal detector: at each step, using only data up to that step, flag
subscriptions whose trailing 12-step growth reaches a threshold.

| threshold | recall | precision | median lead | false alarms/hour |
|---:|---:|---:|---:|---:|
| +2 | 6/10 | **0.69%** | 210 min | 453 |
| +5 | 6/10 | **1.79%** | 198 min | 152 |
| +10 | 5/10 | **3.36%** | 190 min | 60 |

**It does not work as a fleet-wide alert.** At the loosest threshold 874 of
2,324 subscriptions (38%) are flagged before the event. Hundreds of tenants
ramp as hard as the real drivers without moving the aggregate. Per subscription
the rate is low (~0.03/h), so it is usable as a per-tenant signal — but the
capacity policy must not depend on early detection.

---

## 5. Forecasting approach and validation

**Design.** Expanding-window rolling origin inside the training window: fit on
`[0, o)`, forecast `[o, o+H)`. `min_train = max(2H, 4P+2) = 50`, origin step 3,
13 folds at origins 50, 53, …, 86 (the last exactly `N−H`). The first origin is
the shortest window on which the period detector can examine the 2P harmonic,
so every fold answers the seasonality question for the same reason rather than
one fold silently running non-seasonal. Step 3 spreads origins across phases of
the hourly cycle. Folds overlap, so spread is reported as an indicator and no
significance test is used anywhere.

**Baselines:** persistence, last-cycle mean, seasonal-naive, training mean.
**Candidates:** phase profile combined with four different level estimators,
plus damped Holt-Winters.

### 5.1 The decisive validation result

| model | MAE | fold sd | bias | MASE | skill vs persistence |
|---|---:|---:|---:|---:|---:|
| `profile_ar1_train_mean` | 0.1949 | 0.054 | +0.032 | 1.57 | +14.0% |
| **`profile_train_mean`** | 0.1952 | 0.052 | +0.012 | 1.57 | +13.9% |
| `train_mean` | 0.2045 | 0.054 | +0.010 | 1.64 | +9.8% |
| `profile_ar1_recent_mean` | 0.2118 | 0.050 | +0.051 | 1.70 | +6.6% |
| `persistence` | 0.2267 | 0.103 | +0.079 | 1.82 | 0.0% |
| `profile_last_cycle` | 0.2553 | 0.098 | +0.114 | 2.05 | −12.6% |
| `last_cycle_mean` | 0.2615 | 0.096 | +0.114 | 2.10 | −15.3% |
| `seasonal_naive` | 0.2758 | 0.093 | +0.114 | 2.21 | −21.7% |
| `holt_winters` | 0.2875 | **0.241** | +0.195 | 2.30 | −26.8% |

`train_mean` beating every recency-anchored baseline is what shaped the
candidate set: over 24 steps the *level* estimate mattered more than the
seasonal shape. Holt-Winters is best at h ≤ 5 (skill +0.711 at step 1) but
catastrophically unstable (one fold at MAE 0.97).

**Predictability horizon.** Persistence's skill against `train_mean` runs
+0.397, +0.408, +0.251, +0.263, +0.169, +0.021, +0.326, +0.170 over steps 1–8,
first turns negative at **step 9**, and reaches −0.281 by step 24. **Recent
history stops paying after roughly 40 minutes.** The crossing is noisy at this
sample size — skill dips at step 9, recovers briefly at step 10, then stays
negative — so the boundary is approximate rather than a sharp threshold.

### 5.2 Selection rule, amended before freezing

The pre-registered rule — "within one fold-standard-deviation, the simpler
model wins" — failed on contact with the data. That band is built from the
*winner's* MAE spread across folds, which measures how much fold difficulty
varies, not how reliably one model beats another; with overlapping folds it
spanned 0.1949–0.2489 and admitted five of nine candidates, down to
`persistence`. Applied literally it selects `train_mean`.

**Amendment:** the tie-band is one standard error of the **paired** per-fold
MAE difference. Pairing cancels shared fold difficulty, narrowing the band from
0.0539 to 0.0071 and admitting exactly two models. Both rules and their
outcomes are recorded (DECISIONS entry 19), so a reviewer can confirm the
change did not depend on holdout performance.

### 5.3 The profile earned its place; the AR(1) term did not

The two surviving models differed by 0.0002 MAE (0.03 paired se) — unseparable
by accuracy. The tiebreak was a falsifiable mechanistic test: does the profile
remove a bias the mechanism *predicts*?

| check | observed | verdict |
|---|---:|---|
| `train_mean` bias at dip phase 4 (predicted positive) | **+0.2182** | PASS |
| `train_mean` bias at rebound phase 5 (predicted negative) | **−0.1427** | PASS |
| profile reduces dip bias | 0.2182 → **0.0389** (−82%) | PASS |
| profile reduces rebound bias | 0.1427 → **0.0439** (−69%) | PASS |

`corr(profile effect, per-phase bias) = −0.966`: the flat model's error pattern
is almost exactly the negative of the hourly profile.

The AR(1) reversion term was **dropped** despite its nominally lower MAE. Its
fitted φ jumps from ~0.39 in the three earliest folds to ~0.85 in all later
folds, coinciding exactly with the bump entering the training window, and at
φ≈0.85, φ²⁴ ≈ 0.0005 — fully decayed before the horizon ends. A parameter set
by one event, buying no measurable accuracy, was not worth the failure mode.

---

## 6. Holdout results and error analysis

**Submission integrity:** all five checks passed (column order, row count,
timestamps, both target columns matching the pipeline).

The frozen model placed **fourth of five** (table in §1). The holdout mean is
9.2690 against a training mean of 9.4141 — the slide continued — and the
selected model's bias is **+0.145**, over-forecasting at 20 of 24 steps. The
validation ranking **inverted**: `last_cycle_mean` and `seasonal_naive`, which
lost worst across folds, placed first and second.

### 6.1 This was predicted in writing

DECISIONS entry 22, before the gate, from the last four training folds:

| model | MAE last 4 folds | MAE first 9 | bias last 4 |
|---|---:|---:|---:|
| `profile_train_mean` | 0.1839 | 0.2002 | **+0.1839** |
| `persistence` | **0.1213** | 0.2736 | +0.0084 |

Bias equalling MAE means no error changed sign — every prediction was too high
during the post-bump slide. The entry stated: "the holdout immediately follows
the slide, so if the decline continues, the selected model will over-forecast
and `persistence` would have scored better." Acting on that at the time would
have required using knowledge of where the holdout sits in the series, which is
exactly the leakage the protocol forbids.

### 6.2 Separating what failed from what held

- **The profile component held.** The selected model beat `train_mean` 0.1609
  vs 0.1746 (skill +0.079) — the part the gate diagnostic tested did its job.
- **The level estimator failed.** Anchoring on the full training mean assumed
  reversion the data never demonstrated.
- **The root cause is sample size in *regimes*, not in observations.** The
  window contains one bump and one decline. Thirteen folds re-use that single
  episode thirteen times; overlapping folds cannot manufacture a second
  regime. The same single-episode weakness that justified dropping the AR(1)
  term also undermined the mean-anchored level — the inference was applied
  consistently and was still wrong.

### 6.3 Where the evaluation itself is thin

**Peak analysis is vacuous on this holdout.** The training 90th percentile is
9.6684; the holdout maximum is 9.6360. **Zero holdout points qualify as
peaks**, so the required peak metric has no observations. The holdout is the
tail of the decline followed by a mild recovery (+0.017/step, 9.168 → 9.428) —
it contains no high-load period at all. Reported as a gap, not a pass.

Peak behaviour is therefore known only from the training folds, where every
model that won on average under-forecast peaks (bias −0.31 to −0.34) and
`persistence` was best on peaks (MAE 0.1989).

---

## 7. Capacity recommendation

**The band.** Upper bound = forecast + q-quantile of the under-forecast
distribution, per horizon bucket (1–6, 7–12, 13–24), with widths constrained
**non-decreasing** across buckets. The cost assumption lives in the target
breach rate — 10%, i.e. running short costs 9× what idling costs — and the
nominal quantile is *calibrated* to deliver it, because a band built at nominal
0.90 breached 12.5% out of fold. Shipped: nominal q = 0.875, increments
0.364 / 0.368 / 0.368, replayed breach 9.9%.

| q | cost ratio | mean increment | replayed breach | mean idle |
|---:|---:|---:|---:|---:|
| 0.80 | 4× | 0.323 | 15.4% | 0.348 |
| 0.875 | 7× nominal | 0.369 | **9.9%** | 0.389 |
| 0.95 | 19× | 0.463 | 3.2% | 0.477 |

**On the holdout: 0 breaches of 24, mean idle 0.5125 points.** This is *not* a
success. The band held because the point forecast was biased high, so the 0%
rate says nothing about quantile calibration. The informative number is cost:
0.51 points of reserved headroom exceeds the series' entire 0.80-point training
range.

**Recommended policy.**

1. **Provision to the band, not the point forecast**, for two-hour planning.
   At ~9.4% utilisation a 0.37-point margin is operationally free. But be clear
   it is a *forecast-error* margin, smaller than the fleet's own hourly swing —
   not a capacity margin.
2. **Monitor the hot tail, not the mean.** ~2% of VMs run above 50% while the
   fleet sits at 9.4%. Risk is in that tail. Track the share of VMs above 50%
   as the peak-risk trigger; the aggregate forecast cannot see it.
3. **Treat the hourly rebound as scheduled, not stochastic.** The dip→rebound
   swing is 0.342 points, 2.8× a typical step. Subtract the profile before
   alerting, or thresholds will be set ~2× too wide. Use recent history only
   inside the ~40-minute predictability horizon; past that, plan to the level.
4. **Borrow headroom from delay-insensitive capacity** (8.26 mean) before
   interactive (11.51) when tight — a lever invisible in the aggregate.
5. **Forecast core-weighted utilisation.** Capacity is bought in cores. The
   core-weighted target is both more predictable and more cyclical: same model,
   MAE 0.1578 vs 0.1952, MAE/sd 0.891 vs 1.008, profile amplitude 0.4067 vs
   0.3416 (+19%).
6. **Do not build the policy on early detection of demand ramps.** §4.5 shows
   0.7–3.4% precision fleet-wide.

**Under- vs over-provisioning.** At this utilisation the asymmetry is extreme
and favours over-provisioning: idle headroom costs a fraction of a percent of
fleet capacity, while running short on the hot tail risks SLA breaches on the
2% of VMs that actually matter. That is why a 9× cost ratio is defensible here
— and why the same policy would need re-costing on a fleet running at 70%.

---

## 8. Limitations

1. **One 11-hour window, one bump, one decline.** The binding constraint. No
   amount of fold engineering creates a second regime, and this single
   ambiguity caused the model loss.
2. **No calendar structure.** 11.08 hours cannot show daily or weekly
   seasonality, the dominant cycles in real cloud demand. Timestamps are
   trace-relative, so calendar features are not even constructible.
3. **Survivorship and censoring.** Only VMs with complete traces were kept;
   creation is left-censored at 0 and 82% of deletions are right-censored at
   the 30-day trace end. Conclusions describe long-lived VMs observed mid-life.
4. **`active_vm_count` is uninformative** — constant by curation, so its
   perfect score measures nothing.
5. **The holdout contains no peak period**, so the required peak analysis is
   unvalidated out-of-sample.
6. **Band calibration rests on overlapping folds.** Leave-one-fold-out removes
   a fold's own residuals but not the shared history, so breach rates remain
   optimistic and carry no confidence interval. Per-bucket rates are uneven
   (~15% / 14% / 5%).
7. **Attribution is descriptive.** Subscription-level concentration is measured
   over one episode and cannot support a causal scheduling claim.
8. **`">24"` cores mapped to 24**, a stated lower bound that understates 10 VMs.

## 9. Future work

1. **Re-run on a longer trace** — the only fix for limitations 1 and 2, and the
   highest-value next step by a wide margin.
2. **Model the level with a damped local trend rather than a global mean.** The
   failure was specifically an inability to track a slow decline; a damped
   trend hedges between reversion and continuation instead of betting on one.
3. **Adopt core-weighted utilisation as the primary target** (§7.5).
4. **Forecast the hot-tail share directly** as a second target, since that is
   where the operational risk sits.
5. **Segment-level forecasting for the top ~25 subscriptions**, which drive 62%
   of the net hourly cycle.
6. **Revisit the evaluation window.** A single 24-step holdout with no peak
   period cannot discriminate between models; rolling-origin holdout evaluation
   across several windows would.

---

### Reproducing this report

```bash
python -m pytest -q                              # 40 tests
python -m cwfc.predict --data challenge_data --output predictions.csv
python -m cwfc.score_holdout --data challenge_data --predictions predictions.csv
python analysis/01_data_and_workload.py
python analysis/02_cycle_attribution.py
python analysis/03_validation_and_policy.py
```

Supporting detail: [`../DECISIONS.md`](../DECISIONS.md) (31 decisions with
alternatives and outcomes), [`../PLAN.md`](../PLAN.md) (methodology spec),
and the generated reports in this directory.
