# Cloud Workload Forecasting Challenge — submission

Forecasts two-hour-ahead aggregate demand for a 10,000-VM Azure trace, with
the leakage protocol, validation evidence and decision log that justify it.

**Holdout result:** the frozen model scored MAE 0.1609 and placed fourth of
five, behind `last_cycle_mean` (0.1230), `seasonal_naive` (0.1400) and
`persistence` (0.1468); [`DECISIONS.md`](DECISIONS.md) entry 22, written before
scoring, predicted that failure and §7 of
[`reports/final_report.md`](reports/final_report.md) explains it.

---

## 1. Install

Python **3.11+**.

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q                    # 40 tests
```

## 2. Get the data

The dataset is not committed (SUBMISSION.md forbids it, and it is 112 MB).
Download the fixed `data-v1` release and extract it to `challenge_data/`:

```bash
curl -L -o data-v1.tar.gz \
  https://github.com/atomityhq/forecasting-challenge/releases/download/data-v1/data-v1.tar.gz
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
cd challenge_data && sha256sum -c SHA256SUMS && cd ..
```

Expected: `vm_cpu_readings.csv.gz: OK`, `vm_metadata.csv.gz: OK`,
`DATA_CARD.md: OK`.

## 3. Generate predictions

**One command produces the submission:**

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

Writes 24 rows with columns `timestamp,active_vm_count,aggregate_cpu_utilization`,
one per holdout timestamp, in chronological order. Optionally add
`--band-output capacity_band.csv` for the capacity band.

This command **never reads holdout target values** — only the holdout's
`timestamp` column, which the submission must reproduce. Verified by test:
corrupting every holdout reading leaves predictions bit-identical.

## 4. Score the holdout (separate, by design)

```bash
python -m cwfc.score_holdout --data challenge_data --predictions predictions.csv \
    --output reports/holdout_scores.md
```

Scoring is a different command from prediction so that producing a submission
can never print a holdout score — otherwise every development rerun would leak
that score into the author's decisions.

## 5. Reproduce the analysis

```bash
python analysis/01_data_and_workload.py      # -> reports/01_data_and_workload.md
python analysis/02_cycle_attribution.py      # -> reports/02_cycle_attribution.md
python analysis/03_validation_and_policy.py  # -> reports/03_validation_and_policy.md
python analysis/04_core_weighted.py          # -> reports/04_core_weighted.md
python analysis/figures.py                   # -> reports/figures/*.png
```

The prose analysis lives in scripts rather than notebooks (ASSIGNMENT.md allows
"notebooks/ or equivalent analysis"): a script has no hidden execution state,
cannot ship stale outputs beside changed code, and regenerates its markdown on
every run.

[`notebooks/workload_analysis.ipynb`](notebooks/workload_analysis.ipynb) is a
thin viewing layer over the same code — it imports `analysis/figures.py` rather
than re-deriving anything, so it cannot drift from `reports/`. Committed with
outputs so it reads without running.

## 6. Verify a fresh clone reproduces everything

```bash
bash scripts/verify_reproduction.sh
```

Clones into a throwaway directory, follows every README step from a clean
checkout — venv, install, tests, release download, checksum verification,
prediction, scoring, analysis — and asserts the committed `predictions.csv` is
reproduced byte-for-byte. Set `CWFC_RELEASE_CACHE=/path/to/data-v1.tar.gz` to
skip the 112 MB download.

---

## Main modelling decisions

Full reasoning, with alternatives and outcomes, in
[`DECISIONS.md`](DECISIONS.md) (32 entries); the methodology spec is
[`PLAN.md`](PLAN.md), written before modelling and amended as logged.

| Decision | Choice | Why |
|---|---|---|
| Targets | One shared validity predicate (non-null + finite); out-of-range **kept and flagged** | Separates measurement validity from plausibility |
| Missing data | Never filled. Absent grid slot → NA; present-but-unusable → count 0 | Zero and "no observation" mean different things |
| Timestamps | Raw integer seconds, never converted to datetime | They are trace-relative, not wall-clock; no fabricated epoch |
| Seasonality | **Detected per fold**, accepted only if the ACF clears the band at both P and 2P | Separates a real cycle from one spurious correlation |
| Validation | Expanding-window rolling origin, 13 folds, origins 50…86 | First origin = shortest window the detector can test 2P on |
| Selection | Lowest mean MAE; tie-band = one **paired** standard error; simpler wins | Unpaired fold-sd measured fold difficulty, not model difference |
| Selected model | `profile_train_mean` — hourly phase profile + training-mean level | Profile kept because it removes a *predicted* bias, not on MAE |
| Dropped | AR(1) reversion term | Its φ is set by one event and decays before the horizon ends |
| `active_vm_count` | Persistence, declared as config | Constant **by curation**, not a property of real workloads |
| Band | Nominal q **calibrated** to a 10% target breach; widths non-decreasing | Nominal 0.90 actually breached 12.5% out of fold |

### Leakage controls

- Split index derived from sorted unique timestamps; `110` appears nowhere.
- Period, phase profile, MASE scale, peak threshold and band quantiles are all
  fitted **per fold, on that fold's training slice only**.
- Metadata lifetime statistics (`avg_cpu`, `max_cpu`, `p95_max_cpu`,
  `timestamp_vm_deleted`) are descriptive only, never features.
- `predict` and `score_holdout` are separate entry points.
- Model, band and policy frozen in [`src/cwfc/frozen.py`](src/cwfc/frozen.py)
  and tagged `frozen-before-holdout` **before** scoring.

## Layout

```text
src/cwfc/        targets, checks, seasonal, models, baselines, metrics,
                 validation, evaluate, policy, monitor, attribution,
                 frozen (the frozen choices), predict, score_holdout
analysis/        scripts that regenerate reports/ and the figures
notebooks/       thin viewing layer, committed with outputs
scripts/         fresh-clone reproduction check
tests/           40 tests, incl. no-lookahead and a smoke dataset
reports/         generated analysis, figures/ and final_report.md
DECISIONS.md     32 decisions: choice, alternatives, reason, outcome
PLAN.md          methodology spec, written before modelling, amended as logged
predictions.csv  the submission
```

## Not hard-coded

`tests/test_smoke_dataset.py` builds a 200-VM × 60-timestamp dataset with a
**120 s** cadence, seasonal period **6**, deleted rows and an injected
duplicate, then drives the documented `predict` command over it end to end.
Every structural parameter differs from the real release.
