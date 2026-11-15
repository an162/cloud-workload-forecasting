"""Uncertainty bands and the capacity policy they imply (PLAN section 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_BUCKETS = ((1, 6), (7, 12), (13, 24))


def cost_ratio_for_q(q: float) -> float:
    """Under/over cost ratio implied by a band quantile.

    The band quantile follows from the asymmetry of the two mistakes:
    `q = C_under / (C_under + C_over)`, so `C_under / C_over = q / (1 - q)`.
    Quoting the ratio makes the policy's implicit assumption explicit - q=0.9
    is a claim that running short is nine times as expensive as idling.
    """
    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1)")
    return float(q / (1.0 - q))


def label_buckets(preds: pd.DataFrame, buckets=HORIZON_BUCKETS) -> pd.DataFrame:
    """Attach a horizon-bucket label and the under-forecast magnitude."""
    out = preds.copy()
    labels = pd.Series(pd.NA, index=out.index, dtype="object")
    for lo, hi in buckets:
        labels = labels.mask(out["step"].between(lo, hi), f"{lo}-{hi}")
    out["bucket"] = labels
    # error = pred - true, so a shortfall (pred below actual) is positive here.
    out["under_forecast"] = -out["error"]
    return out.dropna(subset=["bucket"])


def under_forecast_quantiles(
    preds: pd.DataFrame, model: str, quantiles=(0.8, 0.9, 0.95), buckets=HORIZON_BUCKETS
) -> pd.DataFrame:
    """Per-bucket quantiles of the under-forecast distribution (pooled folds).

    A negative value means the model over-forecasts so often that even the
    q-th worst shortfall is still a surplus, i.e. the point forecast already
    carries headroom at that confidence level.
    """
    frame = label_buckets(preds[preds["model"] == model], buckets)
    rows = []
    for bucket, group in frame.groupby("bucket", sort=False):
        row = {"bucket": bucket, "n": len(group), "mean_under": group["under_forecast"].mean()}
        for q in quantiles:
            row[f"q{int(q * 100)}"] = float(np.quantile(group["under_forecast"], q))
        rows.append(row)
    return pd.DataFrame(rows)


def replay_policy(
    preds: pd.DataFrame,
    model: str,
    q: float,
    buckets=HORIZON_BUCKETS,
    leave_one_fold_out: bool = True,
    floor_at_forecast: bool = True,
) -> pd.DataFrame:
    """Replay a band policy over the folds and measure what it would cost.

    For each fold the band increment is the q-quantile of the under-forecast
    distribution in that horizon bucket, estimated from the *other* folds when
    `leave_one_fold_out` is set, so a fold is never provisioned using its own
    residuals. Reported per bucket:

    - `breach_rate`    share of points where the actual exceeded the band
      (target: about 1 - q)
    - `mean_idle`      average unused headroom, in CPU points
    - `mean_increment` average band width added above the point forecast

    `floor_at_forecast` clamps a negative increment to zero: provisioning
    *below* one's own forecast is not a capacity policy, so when the forecast
    already over-covers, the band simply adds nothing.
    """
    frame = label_buckets(preds[preds["model"] == model], buckets)
    folds = sorted(frame["fold"].unique())
    rows = []

    for bucket, group in frame.groupby("bucket", sort=False):
        for fold in folds:
            held = group[group["fold"] == fold]
            if held.empty:
                continue
            source = group[group["fold"] != fold] if leave_one_fold_out else group
            if source.empty:
                continue
            increment = float(np.quantile(source["under_forecast"], q))
            if floor_at_forecast:
                increment = max(0.0, increment)
            band = held["y_pred"] + increment
            breach = held["y_true"] > band
            slack = (band - held["y_true"]).clip(lower=0.0)
            rows.append(
                {
                    "bucket": bucket,
                    "fold": fold,
                    "n": len(held),
                    "increment": increment,
                    "n_breach": int(breach.sum()),
                    "sum_idle": float(slack.sum()),
                    "max_shortfall": float((held["y_true"] - band).max()),
                }
            )

    detail = pd.DataFrame(rows)
    summary = detail.groupby("bucket", sort=False).apply(
        lambda g: pd.Series(
            {
                "n": int(g["n"].sum()),
                "mean_increment": float(np.average(g["increment"], weights=g["n"])),
                "breach_rate": float(g["n_breach"].sum() / g["n"].sum()),
                "mean_idle": float(g["sum_idle"].sum() / g["n"].sum()),
                "worst_shortfall": float(g["max_shortfall"].max()),
            }
        ),
        include_groups=False,
    )
    return summary.reset_index()


def policy_table(
    preds: pd.DataFrame,
    model: str,
    quantiles=(0.8, 0.9, 0.95),
    buckets=HORIZON_BUCKETS,
    leave_one_fold_out: bool = True,
) -> pd.DataFrame:
    """The capacity trade-off table: one row per (q, horizon bucket)."""
    out = []
    for q in quantiles:
        replayed = replay_policy(
            preds, model, q, buckets=buckets, leave_one_fold_out=leave_one_fold_out
        )
        replayed.insert(0, "q", q)
        replayed.insert(1, "under_over_cost_ratio", round(cost_ratio_for_q(q), 1))
        replayed["target_breach_rate"] = round(1 - q, 3)
        out.append(replayed)
    return pd.concat(out, ignore_index=True)


def final_band_increments(
    preds: pd.DataFrame, model: str, q: float, buckets=HORIZON_BUCKETS
) -> dict[str, float]:
    """Band increments for the real forecast, from all training folds.

    Used once, at prediction time. Pooling every fold is correct here because
    all of them are training-window data; the leave-one-fold-out variant above
    exists to estimate honestly what this policy *will* cost, not to build it.
    """
    frame = label_buckets(preds[preds["model"] == model], buckets)
    return {
        str(bucket): max(0.0, float(np.quantile(group["under_forecast"], q)))
        for bucket, group in frame.groupby("bucket", sort=False)
    }
