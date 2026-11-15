# PLAN.md — methodology spec (v2, after first data pass)

Notation: `T` = unique timestamps, `H` = holdout length (24, a CLI default),
`N = T - H`, `P` = detected seasonal period (12 on the public data).
All derived at run time. 134, 110 and 12 never appear in code.

---

## 0. What the data told us (training window only, unless structural)

- Complete 10,000 x 134 rectangle: regular 5-min grid, no gaps, no
  duplicates, no nulls, all CPU in [0, 100], min <= avg <= max everywhere.
  The missing-data requirement is answered by a documented proof of absence.
  The cleaning code still exists and is tested, because the smoke dataset
  may not be clean.
- `active_vm_count` is constant at 10,000. This follows from row counts
  alone (rows = VMs x timestamps, zero duplicates), so no holdout value was
  needed to know it. Log this in DECISIONS.md as a structural check.
- Lifecycle fields: `timestamp_vm_created` = 0 for every VM and every
  deletion is after the window. Created = 0 most likely means "already
  running when the trace started" (left-censored), and deletions at the
  maximum value are probably censored at trace end. Check and say so.
  The sample is survivors only: no churn by construction, and long-lived
  VMs may not behave like the whole fleet.
- Aggregate CPU: mean 9.41, std 0.19, total range 0.80 points. Extremely
  stable. Shape: flat ~9.4, a bump to ~9.9 around steps 53-72, then a slow
  slide to ~9.2 (last full-cycle mean ~9.22).
- **Hourly cycle is real.** Diff-ACF 0.61 / 0.58 / 0.41 at lags 12 / 24 /
  36, and it is visible by eye: a dip at step 4, 16, 28, ... 100 followed
  by a rebound one step later, nine cycles in a row. Size ~0.3 points,
  about twice the typical step change.
- Consequence: the lag-1 ACF of 0.66 understates persistence. The sawtooth
  inflates step-to-step noise; the underlying level is smoother.
- Demand is concentrated: top 10% of VMs = 44% of CPU; 2.2% of VMs run
  above 50%. Interactive VMs run hotter (11.5) than delay-insensitive (8.3).

## 1. Targets (settled)

1. Valid reading = `avg_cpu` non-null and finite. Out-of-range values are
   kept and flagged, never clipped silently. Duplicates: identical -> keep
   one; conflicting -> average. Each VM counts once.
2. Grid from min, max and modal step. A timestamp with no rows is NaN for
   both targets, not 0. Target values are never filled.
3. Build on the full series, then split by position: last `H` = holdout.
4. `active_vm_count`: forecast by persistence. Report it as degenerate by
   curation and describe what would be done with real churn (arrivals and
   departures modeled separately). No special case in code.

Extension target (after core work): **core-weighted utilization**,
`sum(avg_cpu * cores) / sum(cores)` using bucket values, with ">24"
treated as 24 (stated lower bound). This is closer to what capacity
planners provision than the unweighted mean.

## 2. Validation (settled)

- Expanding-window rolling origin inside training. Fit on `[0, o)`,
  forecast `[o, o + H)`.
- `min_train = max(2 * H, min_length_for_period(P))`, origin step 3, last
  origin `N - H`. The detector exposes its own minimum: confirming the 2P
  harmonic needs `4P + 2` points, so the first origin is the shortest window
  at which the seasonality question is answerable. Public data: `min_train =
  50`, origins 50, 53, ..., 86 = 13 folds, and all 13 detect P = 12 (at
  `min_train = 48` fold 0 silently ran non-seasonal). Step 3 spreads origins
  across phases of the hourly cycle.
- Folds overlap, so errors are correlated: report mean and spread, no
  significance tests. Too little data for one fold -> fall back to
  persistence with a warning.
- Early folds forecast into the steps 53-72 bump. That is the built-in
  stress test and the centre of the failure analysis.
- Selection rule (**amended before freezing**, DECISIONS entry 19): lowest
  mean MAE over the H-step path; the tie-band is **one standard error of the
  paired per-fold MAE difference** against the leader, and within that band
  the simpler model wins.
  - Superseded rule: the band was one fold-standard-deviation of the
    winner's own MAE. That measures how much fold *difficulty* varies, not
    how reliably one model beats another; because folds overlap it admitted
    5 of 9 candidates (band 0.1949-0.2489, down to `persistence`) and would
    have selected `train_mean`. Pairing cancels the shared fold difficulty
    and narrows the band to 0.0071, admitting exactly two models.
  - No significance claim attaches to the standard error: folds overlap, so
    it is a comparison scale, not a test.
- Freeze, commit, then refit on all `N` and score the holdout once.
- **Frozen:** `profile_train_mean` (`cwfc/frozen.py`), committed before any
  holdout scoring.

## 3. Models (settled)

Seasonality is detected, not assumed: on each training fold, take the
diff-ACF peak lag `P`; accept it only if the ACF at `P` and at `2P` both
exceed the white-noise band. Otherwise run non-seasonal.

Baselines:
1. Persistence (last value). Known weakness: the last point sits at an
   arbitrary phase of the cycle.
2. Last-cycle mean (mean of the last `P` points). Averaging one full cycle
   cancels the seasonality, so this is the fair "level persists" baseline.
3. Seasonal-naive (repeat the last `P` points).
4. Training mean. Reference line for the predictability curve.

Candidates (all share the same shrunk phase profile, fitted per fold):
5. **Profile + last-cycle level**: phase effect = mean deviation from a
   centred P-point moving average, per phase, with shrinkage toward 0 (few
   cycles). Forecast = last-cycle level + phase effect.
5a. **Profile + training-mean level**: same profile, level = full training
   mean. Motivated by the baseline result that `train_mean` beat every
   recency-anchored baseline over the 24-step path.
6b. **Profile + AR(1)-reverting level, `mu` = full training mean**:
   `level_h = mu + phi^h * (level - mu)`, `phi` fitted on the deseasonalized
   series. Tests "does the slide persist or revert".
6c. **Same with `mu` = mean of the last 4P points**: a local reversion
   target, in case the level itself drifts and the full mean is stale.
7. Holt-Winters additive, damped trend, period `P` (statsmodels).
8. Optional: ridge on lags + phase dummies, recursive.

No peak-aware model: peaks are handled by the section 5 band, not by
biasing a model toward them. Peak and average accuracy trade off directly
here (the best-average models under-forecast every peak), and splitting
that trade-off across a model *and* a band would hide it.

Expectation to check, not assume: a profile model wins; the gain over
baseline 2 comes almost entirely from the dip/rebound steps.

**Result (training folds, 13 origins):** `profile_ar1_train_mean` 0.1949
and `profile_train_mean` 0.1952 lead, with `train_mean` 0.2045 third; the
three are within noise of each other on a paired per-fold test (mean
difference 0.0002, sd 0.0258 for the first pair). Decisively worse:
`6c` (mu = last 4P), `seasonal_naive`, everything anchored on the last
cycle, and Holt-Winters (best at h <= 5, but fold-sd 0.24 and one fold at
MAE 0.97). The gain over `train_mean` appears at h >= 13, not in the
dip/rebound steps as expected.

**Reversion evidence rests mostly on one event.** The fitted `phi` jumps
from ~0.38-0.43 in the three earliest folds to ~0.85 in every later fold,
and that jump coincides with the steps 53-72 bump entering the training
window. With `phi ~ 0.85`, `phi^24 ~ 0.0005`, so AR(1) reversion is
complete well before the end of the horizon and `6b` collapses onto `5a`
for most of the path — which is exactly why their scores are
indistinguishable. Any claim that this workload "reverts" is a claim about
one bump and its decay, not a property established across many episodes.

## 4. Metrics

Per model and target, averaged over folds and by horizon step:
- MAE, RMSE, bias.
- Skill vs persistence and vs last-cycle mean: `1 - MAE_model / MAE_base`.
- MASE with scale = in-fold one-step naive MAE; NaN if scale < 1e-9
  (the constant VM count), stated in the report.
- WAPE instead of MAPE.
- Peak performance: peak = actual >= 90th percentile of that fold's
  training values. MAE and bias on those points.
- **Predictability curve**: skill vs the training mean at each step 1..H.
  The step where skill reaches ~0 is the predictability horizon.
- Failure analysis: worst fold, worst step, what the series did there.

## 5. Uncertainty and capacity policy

- Fold residuals pooled into step buckets (1-6, 7-12, 13-24).
- Upper band = forecast + q-quantile of under-forecast error.
- `q = C_under / (C_under + C_over)`. Show q = 0.8 / 0.9 / 0.95 replayed
  over the folds: breach rate and average idle headroom.
- Honest framing: at ~9.4% mean utilization the fleet aggregate is not
  where capacity risk lives. Risk sits in the hot tail (the ~2% of VMs
  above 50%) and in the hourly rebound. Recommend: (a) headroom from the
  band on the aggregate, (b) a segment rule for the VMs that drive the
  hourly cycle, (c) watch the hot-VM share as a peak-risk trigger.
- Caveat: bands come from 13 overlapping folds in one 11-hour window.

## 6. Analyses that make the submission distinctive

1. **Who drives the hourly cycle?** Per-VM contribution to the dip-to-
   rebound jump, averaged over training cycles. Share explained by the top
   N VMs, by category, core bucket, subscription and deployment. If a few
   subscriptions explain most of it, that is a scheduling finding and a
   segment rule.

   **Net and gross are always reported together**, for VM, segment and
   subscription shares alike. Contributions are signed and largely cancel:
   gross movement is ~4x the net move for both the hourly cycle (13827 vs
   3493 summed over VMs; 5815 VMs up, 4185 down) and the bump (14676 vs
   3491; 4727 up, 5273 down). Net shares therefore exceed 100% and can go
   negative, which is arithmetically correct but reads as nonsense on its
   own. Reporting only net would overstate concentration by roughly 3-4x.
2. **Who drove the bump (steps 53-72)?** Same attribution, bump mean minus
   pre-bump mean, net and gross.

   **Early warning is a prospective test, not a retrospective observation.**
   The retrospective version ("the top 1% of bump contributors were already
   ramping") is true but worthless as evidence, because it selects the
   group using the outcome. Replaced by a causal detector: at each step `t`,
   using only data up to `t`, flag any subscription whose trailing P-step
   least-squares growth is at or above a threshold. Score it against a
   retrospective driver label (used for scoring only, never by the
   detector): lead time before step 53, precision, and false alarms per
   monitored hour across the whole training window, at three thresholds.

   **Result: the detector does not work as a fleet-wide alert.** Leads are
   long (95-210 min) but precision is 0.7-3.4% against the top-10-driver
   label, with 60-450 false-alarm episodes per hour across 2324
   subscriptions, because hundreds of subscriptions ramp just as hard
   without moving the aggregate. The honest conclusion is a negative one,
   and the capacity recommendation must not lean on early detection.
3. Survivorship and censoring of lifetimes, stated as a limit on every
   conclusion.
4. Predictability curve (section 4).
5. Core-weighted target (section 1).

## 7. Leakage checklist

- [ ] Split index derived from sorted unique timestamps.
- [ ] Thresholds, percentiles, phase profile, `phi`, `P` fitted on
      training data of the fold only.
- [ ] Rolling features use `shift(1)` before `rolling(...)`.
- [ ] Metadata `avg_cpu`, `max_cpu`, `p95_max_cpu`, `timestamp_vm_deleted`
      are lifetime/future information: descriptive use only.
- [ ] `vm_category` may be derived from lifetime behaviour: descriptive
      segments only, or state the assumption.
- [ ] Analysis that informs modeling uses the training window. Structural
      full-file checks (row counts, validity) are logged as such.
- [ ] `predict` never reads targets past the split; scoring is a separate
      `evaluate` command.
- [ ] Outputs clipped: count >= 0, CPU in [0, 100].

## 8. Tests

- Targets on a hand-made frame with a duplicate, a missing reading, a gap.
- Split and folds work on a 60-timestamp frame; splitter never lets train
  indices reach test indices.
- **No-lookahead**: overwrite the last H target values with garbage, rerun
  predict, assert identical predictions.
- Period detection: finds 12 on a synthetic sawtooth, finds none on noise.
- `predictions.csv` schema: 3 columns in order, H rows, sorted, finite, in
  range, timestamps equal the holdout timestamps.

## 9. Order of work

1. ~~Targets + split + tests~~ DONE.
2. ~~Phase profile and attribution pass (section 6, items 1-2)~~ DONE.
3. ~~Rolling-origin harness, baselines, metrics~~ DONE.
4. ~~Candidates, selection, freeze, commit~~ DONE - `profile_train_mean`
   frozen in `cwfc/frozen.py`, committed before any holdout scoring.
5. Workload analysis notebook with an operational note per finding.
6. ~~Bands + policy table~~ DONE (`cwfc/policy.py`, q=0.9 default).
   Core-weighted extension still open.
7. ~~Predict CLI~~ DONE (`python -m cwfc.predict`). Holdout **not yet
   scored** - that is the next gate, and it happens once.
8. Report (nine sections, 4-6 pages), leakage audit, public link.

### Remaining before the holdout is scored

- README: install, data download, one command to `predictions.csv`.
- `evaluate` command (separate from `predict` by design, §7).

### After scoring, no model may change

Scoring is a one-shot gate. Entry 22 states the expected failure mode in
advance: if the post-bump slide continues, `profile_train_mean` will
over-forecast and `persistence` would have won. Confirming that is a
finding; acting on it afterwards would be leakage.
