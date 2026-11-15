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
