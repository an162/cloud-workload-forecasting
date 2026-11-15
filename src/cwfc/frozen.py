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

from cwfc.models import profile_train_mean

SELECTED_MODEL_NAME = "profile_train_mean"
SELECTED_MODEL = profile_train_mean

HOLDOUT_SIZE = 24
ORIGIN_STEP = 3
PROFILE_SHRINKAGE = 1.0

BASELINE_NAME = "persistence"
REFERENCE_NAME = "train_mean"

FROZEN_AT = "before any holdout scoring; see DECISIONS.md entries 19-22"
