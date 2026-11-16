# Decisions Log

Running record of methodology choices for the Cloud Workload Forecasting Challenge.
Each entry: the choice made, alternatives considered, and the reason.

---

## 1. Environment setup

**Choice:** Standard venv + `pip install -e ".[dev]"` as documented in README.md, no deviation.

**Alternatives considered:** None — this step is prescribed by the scaffold, not a methodology decision.

**Reason:** Matches the reviewer's expected reproduction path exactly (`python -m venv .venv`, `pip install -e '.[dev]'`, `pytest -q`).

**Result:** Python 3.13.9. `pytest -q` → 2 passed (existing `test_metrics.py`, `test_validation.py`). Confirms the scaffold is healthy before any new code is added.

---

## 2. Data acquisition

**Choice:** Downloaded `data-v1.tar.gz` directly via `curl` from the GitHub Releases download URL (public asset, no auth needed), extracted into `challenge_data/`, verified all three files against the bundled `SHA256SUMS`.

**Alternatives considered:** `gh release download` (requires `gh` CLI + auth) — unnecessary since the release asset is public.

**Reason:** Simplest path that still verifies integrity; matches the exact protocol in README.md / DATA_RELEASE.md.

**Result:** All three checksums OK:
```
vm_cpu_readings.csv.gz: OK
vm_metadata.csv.gz: OK
DATA_CARD.md: OK
```

---

## 3. `timestamp` kept as raw int64 seconds

**Choice:** Treat `timestamp` as an opaque, ordered integer grid index (0, 300, ..., 39900). No datetime conversion anywhere in the pipeline.

**Alternatives considered:** Convert to `pd.Timestamp` internally against an arbitrary epoch and convert back for submission — would enable calendar-style feature code paths.

**Reason:** The source timestamps are seconds *relative to the trace*, not wall-clock, so an epoch would be fabricated and hour-of-day/day-of-week features would be meaningless. Keeping the raw integer guarantees `predictions.csv` reproduces the holdout timestamps exactly with no round-trip risk. `rolling_origin_splits` only compares ordering, so it still works.

---

## 4. Cadence inferred as the modal consecutive diff

**Choice:** `cadence_report` infers the expected step as the most common difference between consecutive distinct timestamps, and reports *every* distinct diff with counts alongside it. A gap is any diff exceeding that step; gap count and implied missing-slot count are reported separately.

**Alternatives considered:** (a) minimum diff as the native cadence — survives a majority-irregular series but is fragile to one spuriously close pair; (b) purely descriptive, no inferred canonical step.

**Reason:** Robust to a few gaps while remaining auditable — the full diff distribution is printed next to the inference, so a half-irregular series would be visible rather than hidden behind "step = 300". All three rules agree on this dataset; the choice only matters for a substituted dataset.

**Result:** 134 distinct timestamps, step 300 s (5 min), distinct diffs `{300: 133}`, 0 gaps, 0 missing slots, span 11.083 h. `span / step + 1 == n_timestamps` holds exactly, so the time axis is a perfectly regular grid with no holes.

---

## 5. Both targets defined off one shared validity predicate

**Choice:** A single `valid_reading_mask` (non-null, finite, within `[0, 100]` on `avg_cpu`) defines validity for *both* targets: `active_vm_count` counts distinct VMs passing it, `aggregate_cpu_utilization` averages the values passing it. Invalid/absent readings are excluded, never imputed. A timestamp with nothing valid yields count 0 and a NaN mean rather than a fabricated number.

**Alternatives considered:** Separate predicates per target (e.g. count a VM as active on presence of *any* CPU column, but require a valid `avg_cpu` to enter the mean).

**Reason:** One predicate makes "VM is active at t" and "VM contributes to the mean at t" impossible to diverge, so the two series are always mutually consistent and `aggregate_cpu_utilization(t)` is exactly the mean over the `active_vm_count(t)` VMs. NaN-on-empty keeps missingness visible downstream instead of silently filling.

**Superseded by PLAN.md §1.1 — see entry 8.**

---

## 6. Chronological split derives the training size

**Choice:** `chronological_split(targets, holdout_size)` takes the *trailing* `holdout_size` timestamps as holdout; the training window is `len(targets) - holdout_size` rows, computed never stated. `holdout_size=24` is passed by the caller (the assignment's 2-hour horizon), and 110 appears nowhere in the code.

**Alternatives considered:** Hardcoding the 110/24 boundary or a fixed cut timestamp.

**Reason:** The assignment fixes the *horizon* (24 five-minute steps), not the training length, and reviewers may substitute a smaller dataset. Deriving the cut means a 40-timestamp file splits 16/24 automatically with no code change.

---

## 7. `active_vm_count` established as constant by a structural check (logged per PLAN §0)

**Choice:** The claim "`active_vm_count` is constant at 10,000 for every timestamp" is established from **structural row-count identities over the full file**, not from reading holdout target values:

- 1,340,000 rows / 10,000 distinct `vm_id` / 134 distinct timestamps,
- 0 duplicate `(vm_id, timestamp)` pairs,
- distinct timestamps per VM = 134 for every VM (min = max = 134).

With zero duplicates, `rows = VMs x timestamps` forces every VM to be present exactly once at every timestamp, so the count is 10,000 everywhere. No holdout *value* was needed to reach that conclusion.

**Alternatives considered:** Computing `active_vm_count` on the training window only and leaving the holdout's behaviour formally unknown until scoring.

**Reason:** The panel-completeness check is a data-integrity question about the file, and the assignment requires documenting missingness and duplicates across the supplied data. Deriving the constant from row counts keeps the leakage boundary intact: the fact is structural, available before any modelling, and the training window alone already shows the same constant — so no model choice changes either way.

**Disclosure:** The first EDA pass also printed `active_vm_count.unique()` over all 134 timestamps, which does observe that target's holdout rows. It is recorded here as a deliberate structural check rather than left implicit. It informed no modelling decision; `aggregate_cpu_utilization` holdout values remain unread.

**Consequence (PLAN §1.4):** `active_vm_count` is forecast by persistence and reported as **degenerate by curation** — the DATA_CARD's selection rule kept only VMs with complete 134-observation traces, so zero churn is an artefact of dataset construction, not a property of Azure workloads. No special-casing in code; the harness runs it like any other series and returns MAE 0 for all four baselines, MASE NaN.

---

## 8. Validity = non-null and finite only; range violations flagged, not dropped

**Choice (PLAN §1.1, supersedes entry 5):** A reading is valid iff `avg_cpu` is non-null and finite. Values outside `[0,100]` are **kept and counted separately** via `out_of_range_mask`, never silently dropped or clipped. Duplicate `(vm_id, timestamp)` rows are resolved by averaging, which collapses identical duplicates to the same value and averages conflicting ones in a single step. Clipping to `[0,100]` happens only on the final prediction output.

**Alternatives considered:** Treating out-of-range as invalid (entry 5's original rule) — rejected because an out-of-range value is real source data; excluding it both loses information and silently changes `active_vm_count`.

**Reason:** Separates *measurement validity* (is there a number?) from *plausibility* (is the number in spec?). A reviewer's smoke dataset containing a 150% reading should produce a flagged report, not a quietly shortened series.

**Result on `data-v1`:** 0 invalid, 0 out-of-range, 0 duplicates, 0 absent grid slots — all four cleaning paths are exercised only by tests, which is why the tests matter.

---

## 9. Grid built from min/max/modal step; empty slots stay NA

**Choice (PLAN §1.2):** `full_grid` constructs a complete arithmetic timestamp sequence from min to max using the modal step, rather than using only the timestamps observed in the file. A grid slot absent from the source yields NA for both targets; a slot that has rows but no valid reading yields count 0 and NaN mean.

**Alternatives considered:** Using observed unique timestamps as the index — a missing timestamp would then silently shorten the series and shift every downstream lag.

**Reason:** Makes a gap visibly a gap. The count-0 vs NA distinction is meaningful: 0 means "VMs were present but none reported usably", NA means "no observation exists at all". Neither is ever filled.

---

## 10. Seasonality detected per fold, never assumed

**Choice (PLAN §3):** `detect_period` takes the largest differenced-series ACF peak and accepts it only if the ACF at both that lag **and** twice that lag clear the white-noise band `1.96/sqrt(n)`. It returns `None` otherwise, and models fall back to non-seasonal behaviour (`seasonal_naive` → persistence, `last_cycle_mean` → full-history mean).

**Alternatives considered:** Hardcoding P = 12, or accepting the single highest peak without the harmonic check.

**Reason:** Requiring the harmonic at 2P is what distinguishes a genuine repeating cycle from one spurious correlation, and detecting per fold means the choice is made from training data only. On a reviewer's non-seasonal smoke dataset the same code runs without a seasonal model.

**Result:** P = 12 (60 minutes) detected on the training window and on 12 of 13 folds. Fold 0 (48 training points) returns `None`: the harmonic test needs lag 24 available, which requires at least 50 points. Flagged as an open decision rather than silently patched.

---

## 11. Rolling-origin design: expanding window, positional folds

**Choice (PLAN §2):** Expanding-window origins inside the training window only. `min_train = max(2H, 4P)`, origin step 3, a fold is emitted only while a full H-step path fits. Folds are addressed by **position**, not timestamp.

**Alternatives considered:** The scaffold's `rolling_origin_splits` (timestamp-based, defaults tuned to a 30-day trace) — left intact and still tested, but unusable here because it assumes datetime timestamps and daily horizons.

**Reason:** `2H` guarantees each fold has at least as much history as it must forecast; `4P` guarantees a phase profile is estimable. Step 3 spreads the 13 origins across phases of the 12-step cycle instead of over-sampling one phase. Positional addressing works directly on the raw integer grid and adapts to any dataset length.

**Result:** 13 folds, origins 48, 51, ..., 84, exactly as PLAN predicted. Folds overlap, so metric spread is reported as an indicator with no significance test.

---

## 12. Metric set (PLAN §4)

**Choice:** MAE, RMSE, bias, WAPE, MASE, skill vs persistence and vs last-cycle-mean, per-horizon-step metrics, a predictability curve against the training mean, peak-period errors, and pooled residual quantiles by horizon bucket.

**Alternatives considered:** MAPE (rejected: per-observation division is unstable near zero); a single aggregate metric (rejected by the assignment, which requires more than one).

**Reason:**
- **WAPE instead of MAPE** divides by the aggregate level once, so one near-zero actual cannot dominate.
- **MASE scale = in-fold one-step naive MAE**, returning NaN when that scale is below 1e-9 rather than infinity — the constant `active_vm_count` series hits exactly this case, and NaN states "undefined" honestly.
- **Bias** separates over- from under-forecasting, which is what the capacity policy actually trades off.
- **Peak threshold = that fold's training 90th percentile**, computed per fold from training data only, so peak definition can never leak.

---

## 13. First origin set by the detector's own minimum (decision 1, option b)

**Choice:** `min_train = max(2H, min_length_for_period(P))`, where `seasonal.min_length_for_period(P) = 4P + 2` is the shortest series on which `detect_period` can examine lag 2P. The detector exposes that minimum rather than callers guessing it.

**Alternatives considered:** (a) accept fold 0 running non-seasonal and document the fallback; (c) relax the rule to accept the primary ACF peak when 2P is unavailable.

**Reason:** Under the old `max(2H, 4P) = 48`, fold 0 had 48 points and silently returned `None`, so one of 13 folds answered the seasonality question differently *because of a detector limitation rather than the data*. Deriving the floor from the detector means every fold makes the same decision for the same reason, and the constant lives in the detector that owns it instead of being duplicated in the fold planner. Option (c) was rejected because dropping the harmonic requirement weakens exactly the check that separates a real cycle from one spurious correlation.

**Result:** `min_train = 50`, origins 50, 53, ..., 86 — still 13 folds, and the last origin now lands exactly on `N - H = 86`. All 13 folds detect P = 12.

---

## 14. Candidate set extended; selection rule unchanged (decision 2)

**Choice:** Added three candidates to PLAN §3, all sharing the same per-fold shrunk phase profile: `profile_train_mean` (level = full training mean), `profile_ar1_train_mean` (AR(1) reversion toward the full training mean), and `profile_ar1_recent_mean` (same, reverting toward the mean of the last 4P points). Kept candidate 5 (`profile_last_cycle`) and Holt-Winters.

**Alternatives considered:** Dropping candidate 5 given that `last_cycle_mean` already lost to `train_mean` among the baselines.

**Reason:** `train_mean` beating every recency-anchored baseline said the *level* estimate was what mattered, not the seasonal shape — so the candidate set had to vary the level estimator (last cycle / full mean / reverting to full mean / reverting to recent mean) while holding the profile fixed. Keeping candidate 5 preserves the controlled comparison that isolates the level choice.

**Result:** `profile_ar1_train_mean` 0.1949 and `profile_train_mean` 0.1952 lead; `train_mean` 0.2045. `profile_ar1_recent_mean` (0.2118) is decisively worse on a paired test (mean/se = 3.5), so the *global* mean is the better reversion target. Holt-Winters is best for h <= 5 but has fold-sd 0.2414 and one fold at MAE 0.9715.

**Open issue — the selection rule is ambiguous at this noise level.** "Within one fold-standard-deviation, the simpler model wins" admits five models (band 0.1949 + 0.0539 = 0.2489, including `persistence`), because the fold-sd is large relative to the gaps between models. Flagged for decision rather than resolved unilaterally.

---

## 15. Peaks handled by the band only (decision 3)

**Choice:** No peak-aware model and no peak term in model selection. The peak-vs-average tension is handled entirely by the §5 uncertainty band.

**Alternatives considered:** Carrying a peak-aware model into selection, or adding a peak-weighted term to the selection metric.

**Reason:** Every model that wins on average under-forecasts every peak (peak bias −0.31 to −0.34 for the leaders). That is a *decision* trade-off, not a modelling defect: splitting it across both a model and a band would obscure where the safety margin comes from and make the policy's cost assumption implicit. Keeping the point forecast unbiased-on-average and putting all the asymmetry in one explicit, costed quantile keeps the trade-off auditable.

---

## 16. Attribution reports net and gross together (decision 4)

**Choice:** `contribution_shares` and `top_vm_shares` return both net (signed) and gross (absolute) shares, plus the up/down split and a cancellation ratio, for VM, segment **and** subscription views of both the hourly cycle and the bump.

**Alternatives considered:** Net only (the original pass), or normalising net shares so they sum to 100%.

**Reason:** Gross movement is ~4x the net move in both events — the aggregate is a small residual of large offsetting VM-level activity (hourly cycle: 5815 VMs up, 4185 down; bump: 4727 up, 5273 down). Net-only reporting produced cumulative shares above 100% (top 25 subscriptions = 142% of the bump) which look like errors, and it overstates concentration by 3-4x: the top 10 subscriptions are 98.3% of the *net* bump but only 28.3% of the *gross* movement. Both numbers are true and they support different operational claims, so neither can be dropped.

---

## 17. Early warning replaced by a prospective detector (decision 5)

**Choice:** The retrospective claim is withdrawn. In its place, a causal detector: at each step `t`, using only data up to `t`, flag any subscription whose trailing P-step least-squares growth is at or above a threshold. Scored at thresholds 2 / 5 / 10 CPU points per hour against two retrospective driver labels (top-10 by net bump share; >= 1% net bump share), reporting lead time, precision, and false-alarm episodes per monitored hour across the entire training window.

**Alternatives considered:** Keeping the retrospective statement with a caveat.

**Reason:** The retrospective version selects its group *using the outcome*, so "they were already ramping" is guaranteed by construction and carries no predictive information. Only a test that could have been run at time `t` can support an operational claim.

**Result — a negative finding, reported as such.** Leads are long (median 190-210 min, i.e. the drivers were ramping before monitoring could even start) but precision is **0.7%-3.4%** against the top-10 label with **60-450 false-alarm episodes per hour** over 2324 subscriptions; at threshold 2, 874 of 2324 subscriptions (38%) are flagged before step 53. Hundreds of subscriptions ramp as hard as the real drivers without moving the aggregate. The capacity recommendation must not lean on early detection. The label is retrospective and used for scoring only; the detector never sees it.

---

## 18. Reversion evidence rests mostly on one event (decision 6)

**Choice:** Recorded as an explicit limitation on every reversion-based claim, in both PLAN §3 and the report's limitations section.

**Reason:** The fitted `phi` is **unstable across folds in a way that tracks a single event**: ~0.3886 / 0.3814 / 0.4289 in the three earliest folds, then ~0.70 and ~0.85 in every later fold, with the jump coinciding with the steps 53-72 bump entering the training window. And because `phi ~ 0.85` gives `phi^24 ~ 0.0005`, the AR(1) term has fully decayed well before the end of the 24-step horizon, so `profile_ar1_train_mean` collapses onto `profile_train_mean` over most of the path — which is exactly why their scores differ by 0.0002 (paired sd 0.0258, 7 of 13 folds favouring the AR(1) version).

**Consequence:** "This workload mean-reverts" is a statement about one bump and its decay observed in an 11-hour window, not a property established across repeated episodes. The AR(1) term is retained only because it costs nothing, not because the data demonstrates reversion.

---

## 19. AMENDMENT to the selection rule (supersedes the rule frozen in PLAN §2)

**The old rule (as written in PLAN §2, decided before any candidate was run):**

> lowest mean MAE over the H-step path; within one fold-standard-deviation, the simpler model wins.

**Why it failed.** The band is built from the *winner's standard deviation across folds*, which measures how much fold difficulty varies — not how reliably one model beats another. On this data that spread (0.0539) is five to twenty times the gaps between models, so the band `[0.1949, 0.2489]` admitted five of nine candidates, down to and including `persistence` at 0.2267. Applied literally, **the old rule selects `train_mean`** (the simplest model inside the band; on an aggressive reading of "simpler", `persistence`). The rule was not wrong in spirit — it was measuring the wrong quantity, because folds overlap and share data, so per-model MAEs are strongly correlated and their marginal spread says nothing about a head-to-head comparison.

**The new rule:** lowest mean MAE over the H-step path; the tie-band is **one standard error of the paired per-fold MAE difference against the leader**, and within that band the simpler model wins.

**Why it is better.** Pairing differences fold-by-fold cancels the shared fold difficulty that inflated the old band, which is the standard correction when comparing models on overlapping evaluation sets. The band narrows from 0.0539 to 0.0071 and admits exactly two models (`profile_ar1_train_mean` 0.1949, `profile_train_mean` 0.1952; mean paired difference 0.0002, 7 of 13 folds favouring each). No significance claim is attached to the standard error — folds overlap, so it is a comparison scale, not a test (consistent with PLAN §2's no-significance-tests stance).

**What it selects:** `profile_train_mean`, by the unchanged "simpler wins" clause.

**Amendment status:** documented amendment, decided *before* the holdout was touched and recorded with the superseded rule and its outcome, so the change is auditable rather than retrofitted. Both rules' selections are stated above so a reviewer can see the choice did not depend on holdout performance.

---

## 20. AR(1) reversion term dropped

**Choice:** `profile_ar1_train_mean` and `profile_ar1_recent_mean` are removed from the candidate set. The selected model carries no reversion term.

**Alternatives considered:** Retaining the AR(1) variant since it had the (statistically indistinguishable) lowest mean MAE.

**Reason:** Its apparent edge is 0.0002 MAE — 0.03 standard errors — and entry 18 shows the fitted `phi` is unstable in a way that tracks the single steps 53-72 bump (0.38-0.43 in the three earliest folds, ~0.85 thereafter). With `phi ~ 0.85`, `phi^24 ~ 0.0005`, so the term is fully decayed before the horizon ends and contributes almost nothing beyond the first few steps. Keeping a parameter whose value is set by one event, in exchange for no measurable accuracy, adds a failure mode and an unsupportable claim ("this workload reverts") for nothing.

**Note:** `models.ar1_phi` is retained as a **diagnostic only**, not used by any model, so the `phi`-instability evidence behind entry 18 stays reproducible.

---

## 21. Gate diagnostic: the phase profile is kept on mechanistic grounds

**Choice:** Before freezing, the profile had to earn its place by removing a *predicted* bias, not by winning on MAE. Required: `train_mean` shows the expected signed bias at the dip and rebound phases, and the profile removes it. Otherwise `train_mean` would have been selected.

**Result — passed on all four checks** (error = pred − true, so positive = over-forecast):

| check | expected | observed | verdict |
|---|---|---|---|
| `train_mean` bias at dip phase 4 | positive (over-forecasts the dip) | **+0.2182** | PASS |
| `train_mean` bias at rebound phase 5 | negative (under-forecasts the rebound) | **−0.1427** | PASS |
| profile reduces dip bias | smaller magnitude | 0.2182 → **0.0389** (−82%) | PASS |
| profile reduces rebound bias | smaller magnitude | 0.1427 → **0.0439** (−69%) | PASS |

Supporting evidence: `corr(profile effect, train_mean per-phase bias) = −0.966`, i.e. the mean model's per-phase error pattern is almost exactly the negative of the hourly profile — precisely what "a flat model is blind to the within-hour shape" predicts. Per-phase bias spread falls from sd 0.1010 to 0.0454, mean absolute per-phase bias from 0.0754 to 0.0416, and MAE at the dip/rebound phases from 0.2310 to 0.1654.

**Reason for gating this way:** two models within 0.0002 MAE cannot be separated by accuracy, so the tiebreak must be whether the extra component corrects an error the mechanism predicts. That is a falsifiable test the profile could have failed.

---

## 22. LIMITATION: the selected model's known failure mode is a sustained trend

**Finding (training folds only; recorded for the limitations section, explicitly excluded from selection):** over the last four folds (origins 77, 80, 83, 86 — the post-bump slide, slope −0.005 to −0.007 per step), the mean-anchored models over-forecast *every single point*:

| model | MAE last 4 folds | MAE first 9 | bias last 4 | bias first 9 |
|---|---:|---:|---:|---:|
| `profile_train_mean` | 0.1839 | 0.2002 | **+0.1839** | −0.0650 |
| `train_mean` | 0.1843 | 0.2135 | +0.1823 | −0.0665 |
| `persistence` | **0.1213** | 0.2736 | +0.0084 | +0.1103 |

Bias equal to MAE means no error changed sign. Fold-sequence bias runs −0.289 (during the bump rise) to +0.211 (during the slide): a mean-anchored model lagging a trend in both directions.

**Why this matters and why it did not change the selection:** the holdout immediately follows the slide, so if the decline continues, the selected model will over-forecast it and `persistence` would have scored better. Acting on that would mean selecting a model using knowledge of where the holdout sits in the series, which is exactly the leakage the protocol forbids. The honest position is to select on the training evidence, state this failure mode in advance, and let the holdout confirm or refute it. For capacity purposes the direction is the safe one (over-forecasting adds headroom), which is noted in the policy section.

---

## 23. Band estimated leave-one-fold-out, applied pooled

**Choice (PLAN §5):** Residuals are pooled into horizon buckets (1-6, 7-12, 13-24) and the band increment is the q-quantile of the *under-forecast* distribution (`actual - forecast`). Two estimation modes, used for different purposes:

- **Leave-one-fold-out** to *measure* what the policy will cost: each fold's band comes from the other 12 folds, so no fold is ever provisioned using its own residuals.
- **Pooled over all folds** to *build* the band actually shipped, since every fold is training-window data.

**Alternatives considered:** Pooled estimation for both. Rejected: it reports a breach rate that the policy will not achieve in production.

**Reason:** PLAN §7's leakage rule ("percentiles fitted on training data of the fold only") applies to the band as much as to a model. The difference is measurable and in the expected direction — pooled estimation flatters the policy:

| q | target breach | pooled (in-sample) | leave-one-fold-out |
|---:|---:|---:|---:|
| 0.80 | 0.200 | 0.205 | **0.228** |
| 0.90 | 0.100 | 0.103 | **0.125** |
| 0.95 | 0.050 | 0.051 | **0.077** |

**Also decided:** a negative increment is floored at zero. At long horizons the model over-forecasts often enough that the 80th-percentile shortfall is already a surplus (bucket 13-24, q=0.8: +0.004). Provisioning *below* one's own forecast is not a capacity policy, so the band adds nothing there rather than subtracting.

---

## 24. Capacity policy table and the honest framing

**Choice:** `q = C_under / (C_under + C_over)` is reported as an explicit cost-ratio claim rather than a confidence level, at q = 0.8 / 0.9 / 0.95 (under-provisioning 4x / 9x / 19x as costly as idling).

**Result (leave-one-fold-out):**

| q | ratio | bucket | increment | breach rate | mean idle |
|---:|---:|---|---:|---:|---:|
| 0.80 | 4x | 1-6 | 0.321 | 0.244 | 0.253 |
| 0.80 | 4x | 7-12 | 0.320 | 0.256 | 0.308 |
| 0.80 | 4x | 13-24 | 0.014 | 0.205 | 0.145 |
| 0.90 | 9x | 1-6 | 0.408 | 0.115 | 0.329 |
| 0.90 | 9x | 7-12 | 0.383 | 0.115 | 0.359 |
| 0.90 | 9x | 13-24 | 0.184 | 0.135 | 0.292 |
| 0.95 | 19x | 1-6 | 0.463 | 0.064 | 0.377 |
| 0.95 | 19x | 7-12 | 0.422 | 0.077 | 0.394 |
| 0.95 | 19x | 13-24 | 0.350 | 0.083 | 0.447 |

**Reason for the framing:** at q=0.9 the whole band is **0.18-0.41 CPU points, i.e. 1.9%-4.3% of a 9.41% mean utilisation**. Quantifying it this way is what justifies PLAN §5's conclusion: the fleet aggregate is nowhere near a capacity limit, so the band is cheap but also not where the risk lives. The band is a *forecast-error* margin, not a capacity margin. The default is **q = 0.9** (`frozen.BAND_QUANTILE`), chosen as a stated cost assumption, not fitted.

**Caveat carried into the report:** these bands come from 13 overlapping folds inside a single 11-hour window, and the long-horizon bucket's increment is small only because the model over-forecasts during the post-bump slide (entry 22). A different regime would move it.

---

## 25. `predict` and `evaluate` are separate commands

**Choice:** `python -m cwfc.predict --data challenge_data --output predictions.csv` produces the submission. It reads the holdout's `timestamp` column and nothing else from the holdout. Scoring is a separate command, written only when the holdout is actually scored.

**Alternatives considered:** One command that predicts and reports accuracy.

**Reason (PLAN §7):** if producing a submission also printed a holdout score, every rerun during development would leak that score into the author's decisions. Separating them makes the leakage boundary a property of the code, not of the author's discipline. The submission contract (column order, row count, bounds, integer counts) lives in `frozen.SUBMISSION_COLUMNS` / `OUTPUT_BOUNDS` so the writer cannot drift from SUBMISSION.md.

**Per-target models as configuration:** `frozen.MODEL_BY_TARGET` maps each target to its model (`aggregate_cpu_utilization` → `profile_train_mean`, `active_vm_count` → `persistence`). This honours PLAN §1.4's "no special case in code": the constant target goes through identical machinery and gets persistence because that is its declared model, not because the pipeline branches on its name.

---

## 26. Leakage verified by test, not by inspection

**Choice:** A no-lookahead test overwrites every holdout CPU reading with garbage, reruns prediction, and asserts bit-identical output — at the function level and through the CLI, across four corruptions (999.0, -1.0, 0.0, NaN).

**Reason:** "I checked that the code does not use the holdout" is an assertion about author attention; a test that corrupts the holdout and demands identical predictions is a property the build enforces on every run. Confirmed on the real `data-v1` release as well, with max absolute difference 0.00e+00 for all four corruptions.

**Suite now at 36 tests**, covering PLAN §8: target construction against a frame carrying identical duplicates, conflicting duplicates, a null reading, a mid-series gap, an out-of-range value, and a timestamp present with no valid reading; the zero-versus-NA distinction; split sizes derived rather than stated; folds never letting train indices reach test indices; the fold planner respecting the detector's minimum; period detection finding the planted period on a sawtooth and nothing on noise or a pure trend; profile centring and shrinkage; and the submission schema (exact column order, row count, bounds, integer counts, correct timestamps, header text).

**One limitation pinned deliberately as a test:** a gap cannot be inferred from two timestamps, because the modal step of `[0, 600]` *is* 600. `test_gap_is_not_inferable_from_two_timestamps` documents this rather than asserting away a real property of the grid inference.

---

## 27. Band: nominal quantile calibrated, widths constrained non-decreasing

**Choice:** Three changes to the band, all before the holdout gate.

1. **The cost claim moved from the nominal quantile to the target breach rate.** `frozen.TARGET_BREACH_RATE = 0.10` encodes the 9x under/over cost ratio. The nominal quantile fed to the band is *derived* by leave-one-fold-out replay (`policy.select_nominal_q`) as the smallest q achieving that breach rate.
2. **Widths constrained non-decreasing across horizon buckets** (running maximum in horizon order).
3. **The overlap caveat stated explicitly** in `replay_policy`'s docstring and here.

**Reason for (1):** a band built at nominal q=0.90 breached 12.5% of the time out of fold, not 10%. Fixing the nominal quantile therefore advertises a cost ratio the policy does not deliver. Calibrating against the replayed breach rate makes the stated assumption the thing that actually holds.

**Reason for (2):** the raw quantiles *fall* with horizon (0.407 / 0.385 / 0.178 at q=0.90), which would claim a two-hour-ahead forecast needs less headroom than a thirty-minute one. That is an artefact of the selected model increasingly over-forecasting during the post-bump slide (entry 22), not a property of forecast uncertainty, which cannot shrink as the horizon grows. The constraint encodes that prior and is deliberately conservative.

**Interaction, worth noting:** the two changes push in opposite directions. Monotonicity alone widened the long-horizon band from 0.178 to 0.407 and dropped the breach rate at nominal 0.90 from 12.5% to 7.4% — now *over*-covering. So the calibrated nominal q came out **below** 0.90, at **0.875**, achieving an overall replayed breach rate of **9.9%**.

**Shipped band:** nominal q = 0.875, increments 0.364 / 0.368 / 0.368 points across buckets 1-6 / 7-12 / 13-24.

| bucket | increment | replayed breach | mean idle |
|---|---:|---:|---:|
| 1-6 | 0.367 | 0.154 | 0.290 |
| 7-12 | 0.371 | 0.141 | 0.347 |
| 13-24 | 0.371 | 0.051 | 0.459 |
| **overall** | 0.369 | **0.099** | 0.389 |

**Known unevenness, not hidden:** the overall rate hits the 10% target but the per-bucket rates do not — short horizons breach ~15% while the long bucket breaches 5%. One quantile plus a monotone constraint cannot equalise all three. Per-bucket calibration would fix it at the cost of three parameters fitted on 78-156 points each; left as a single parameter deliberately, with the unevenness reported.

**Caveat (3):** leave-one-fold-out removes a fold's own residuals but **not the overlap between folds** — expanding-window origins share every earlier observation, so a held-out fold's errors stay correlated with the residuals used to build its band. These breach rates are less optimistic than in-sample ones but are still not out-of-sample estimates in the independent-sample sense, and carry no confidence interval. All of it rests on 13 overlapping folds inside a single 11-hour window.

---

## 28. Core-weighted utilisation reported as an extension (PLAN §1)

**Choice:** `build_core_weighted_target` computes `sum(avg_cpu * cores) / sum(cores)` per timestamp, with `virtual_core_bucket` mapped to its numeric value and `">24"` mapped to **24 as a stated lower bound**. Run through the harness with the frozen model and all declared baselines. **Report only — it does not replace the required target and did not influence the frozen choice.**

**Reason:** the required `aggregate_cpu_utilization` weights every VM equally, so a 2-core VM at 90% counts as much as a 24-core VM at 10%. Capacity is bought in cores, so this is closer to the quantity a planner provisions.

**Result — the capacity-relevant target is both more predictable and more cyclical:**

| | core-weighted | unweighted |
|---|---:|---:|
| mean | 9.3131 | 9.4141 |
| sd | 0.1770 | 0.1936 |
| correlation with unweighted | 0.9027 | 1.0 |
| detected period | 12 | 12 |
| **profile amplitude** | **0.4067** | 0.3416 |
| frozen-model MAE | **0.1578** | 0.1952 |
| frozen-model WAPE | 1.70% | 2.06% |
| skill vs persistence | **+0.178** | +0.139 |
| MAE / series sd | **0.891** | 1.008 |

Total cores are constant at 28,316 and all 10,000 VMs matched metadata, so no weighting mass was lost. The hourly cycle is **19% larger** in core-weighted terms, meaning the VMs driving it are above-average in size — consistent with entry 16's finding that the `>24`-core bucket had a net lift of 17x on the hourly jump. The frozen model also ranks *best on peaks* here (peak MAE 0.218 vs 0.265 for `train_mean`), unlike on the unweighted target where it sat mid-pack.

**Operational reading:** if the team provisions cores, this target is the better one to forecast — same model, ~19% lower error relative to its own variability, and a stronger signal to exploit. Flagged as the strongest candidate for follow-up work.

---

## 29. Smoke-dataset test: the pipeline is not wired to the real release

**Choice:** `tests/test_smoke_dataset.py` builds a ~200-VM x 60-timestamp dataset whose every structural parameter differs from the real release — 120 s cadence (not 300), seasonal period 6 (not 12), 60 timestamps (not 134), 200 VMs (not 10,000) — with 7 VM-timestamp rows deleted and one conflicting duplicate injected, then drives the documented `python -m cwfc.predict` command over it end to end.

**Reason:** SUBMISSION.md states reviewers may swap in a tiny internal dataset to check the pipeline is not hard-coded. Rather than assert that it is not, this runs the real entry point on data that would break any hard-coded assumption, and checks the output contract holds.

**What it pins:**
- the injected defects are *detected* (1 conflicting duplicate group, 2 duplicate rows, step inferred as 120);
- deleted rows make `active_vm_count` genuinely vary (min < 200 = max, more than one distinct value), so the degenerate-constant case is not assumed anywhere;
- the submission contract holds on the new data: exact column order, 24 rows, correct timestamps, finite, in range;
- predicted values track the smoke data's own scale (10-35%), not the release's ~9.4%;
- the core-weighted target builds, with `">24"` mapped to 24;
- **a capacity band refuses to ship when no validation folds fit** — with 36 training points no fold supports a 24-step horizon, and `capacity_band` raises rather than inventing a band with nothing behind it.

---

## 30. HOLDOUT SCORED (one shot, after the freeze) — the predicted failure occurred

**Command:** `python -m cwfc.score_holdout --data challenge_data --predictions predictions.csv`, run once at tag `frozen-before-holdout`. Submission integrity passed on all five checks (column order, row count, timestamps, and both target columns matching what the pipeline produces).

**`aggregate_cpu_utilization` — the frozen model placed fourth of five.**

| model | MAE | RMSE | bias | WAPE | MASE | skill vs persistence |
|---|---:|---:|---:|---:|---:|---:|
| last_cycle_mean | **0.1230** | 0.1613 | −0.053 | 1.33% | 1.016 | +0.162 |
| seasonal_naive | 0.1400 | 0.1590 | −0.053 | 1.51% | 1.156 | +0.046 |
| persistence | 0.1468 | 0.1848 | −0.104 | 1.58% | 1.212 | 0.000 |
| **profile_train_mean** (selected) | **0.1609** | 0.1941 | **+0.145** | 1.74% | 1.328 | **−0.096** |
| train_mean | 0.1746 | 0.2105 | +0.145 | 1.88% | 1.442 | −0.189 |

**This is entry 22's failure mode, realised.** That entry, written before the gate, said: "the holdout immediately follows the slide, so if the decline continues, the selected model will over-forecast it and `persistence` would have scored better." The holdout mean is 9.2690 against a training mean of 9.4141 — the slide did continue — and the selected model's bias is **+0.145**, over-forecasting at 20 of 24 steps. The fold ranking inverted almost exactly: the two models that lost worst in training validation (`last_cycle_mean`, `seasonal_naive`) placed first and second.

**No changes follow.** The selection stands as made, on the training evidence available at the time, with the failure stated in advance. Acting on this result would be the leakage the protocol exists to prevent. What it buys is a calibrated statement about *why* the method failed, which is worth more than a better number obtained by cheating.

**What it says about the methodology, honestly:**
- The paired-se tie-band (entry 19) picked between two models that were genuinely indistinguishable, and that part held: `profile_train_mean` and the dropped AR(1) variant would have scored within noise of each other here too.
- The *level* estimator was the consequential choice, and the training folds pointed the wrong way. Thirteen overlapping folds inside one 11-hour window could not distinguish "mean-reverting" from "slowly declining", because only one episode of either was observable. Entry 18 already flagged that the reversion evidence rested on one event; the holdout shows the same single-episode problem also undermined the *mean-anchored* choice.
- The gate diagnostic (entry 21) was still right about what it tested: the phase profile does remove dip/rebound bias. The selected model beat `train_mean` on the holdout (0.1609 vs 0.1746, skill +0.079), so the profile component helped. The level it was bolted onto is what lost.

**`active_vm_count`:** all five models MAE 0.0000, MASE NaN, skill NaN — exactly as predicted for a target that is constant by curation (entry 7). The holdout is 10,000 at all 24 timestamps.

**Peak analysis is vacuous on this holdout:** the training 90th percentile is 9.6684 and the holdout maximum is 9.6360, so **zero holdout points qualify as peaks**. The peak-period analysis therefore has no observations and the §4 peak metric is undefined here. Reported as a gap in the evaluation rather than a passing grade. The holdout contains no high-load period at all — it is the tail of the decline, and then a mild recovery (slope +0.017/step over the 24 steps, from 9.168 up to 9.428).

---

## 31. Capacity band on the holdout: zero breaches, and that is not a success

**Result:** 0 breaches of 24 (0.0%) against a 10% target, with mean idle headroom 0.5125 points.

| bucket | increment | breaches | breach rate | mean idle |
|---|---:|---:|---:|---:|
| 1-6 | 0.364 | 0 | 0.0% | 0.644 |
| 7-12 | 0.368 | 0 | 0.0% | 0.582 |
| 13-24 | 0.368 | 0 | 0.0% | 0.412 |

**Why this is not a win.** The band held for the same reason the point forecast lost: the model over-forecast throughout, so the actual never came near the upper bound. A band whose point forecast is biased high will not breach regardless of whether its width is right. The 0% rate is therefore uninformative about band calibration — it confirms the direction of the bias, not the quality of the quantile estimate.

**The cost side is the real reading.** 0.51 points of mean idle headroom on a series whose total training range is 0.80 points means the policy reserved more headroom than the entire observed variation of the signal. At 9.3% mean utilisation that is operationally free, which is the honest framing from PLAN §5 — but on a tighter-running fleet this would be visible waste, and the monotone constraint (entry 27) contributed to it by widening the long-horizon bucket from 0.178 to 0.368.

**Carried into the report as a limitation, not a finding:** one 24-step holdout containing zero peak points cannot validate a capacity band. The band's calibration evidence remains the 13 overlapping training folds, with the overlap caveat from entry 27 intact.
