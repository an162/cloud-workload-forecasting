"""Frozen modelling choices, fixed before the holdout was ever scored.

Everything in this module was decided from the 110-step training window using
expanding-window rolling origin validation (DECISIONS entries 11-22). Nothing
here may be revised in response to holdout performance; a later change would
be a new, dated amendment with its own entry.

The selection, in brief:

- Candidates scored over 13 expanding-window folds (origins 50..86).
- `profile_train_mean` and `profile_ar1_train_mean` finished 0.0002 MAE apart
  (0.03 paired standard errors) and were the only two inside the tie-band.
- The AR(1) term was dropped: its fitted reversion rate is set by one event
  and has fully decayed before the horizon ends (entries 18, 20).
- The phase profile was kept only because it removes a *predicted* signed
  bias at the dip and rebound phases (entry 21), not on its MAE.

Known failure mode, stated in advance (entry 22): a mean-anchored level
over-forecasts a sustained decline. In the last four training folds every
single prediction was too high (bias == MAE == +0.18).
"""

from __future__ import annotations

from cwfc.baselines import persistence
from cwfc.models import profile_train_mean
from cwfc.targets import ACTIVE_VM_COUNT, AGGREGATE_CPU

SELECTED_MODEL_NAME = "profile_train_mean"
SELECTED_MODEL = profile_train_mean

#: Which model forecasts which target. Declared as configuration rather than
#: a branch inside the prediction code, so `active_vm_count` is handled by the
#: same machinery as everything else (PLAN 1.4: "no special case in code").
#: It gets persistence because it is constant by curation, not because the
#: pipeline knows anything special about it.
MODEL_BY_TARGET = {
    AGGREGATE_CPU: profile_train_mean,
    ACTIVE_VM_COUNT: persistence,
}

MODEL_NAME_BY_TARGET = {
    AGGREGATE_CPU: "profile_train_mean",
    ACTIVE_VM_COUNT: "persistence",
}

#: Output domains enforced on the final predictions (PLAN 7, SUBMISSION.md).
#: `active_vm_count` is additionally rounded to a whole number of VMs.
OUTPUT_BOUNDS = {
    AGGREGATE_CPU: (0.0, 100.0),
    ACTIVE_VM_COUNT: (0.0, None),
}

INTEGER_TARGETS = (ACTIVE_VM_COUNT,)

#: Exact column order required by SUBMISSION.md. Declared here so the writer
#: cannot drift from the contract.
SUBMISSION_COLUMNS = ("timestamp", ACTIVE_VM_COUNT, AGGREGATE_CPU)

#: Band quantile for the capacity policy. q = C_under / (C_under + C_over),
#: so 0.9 asserts that running short costs nine times what idling costs.
BAND_QUANTILE = 0.9

HOLDOUT_SIZE = 24
ORIGIN_STEP = 3
PROFILE_SHRINKAGE = 1.0

BASELINE_NAME = "persistence"
REFERENCE_NAME = "train_mean"

FROZEN_AT = "before any holdout scoring; see DECISIONS.md entries 19-22"
