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


def bucket_labels(buckets=HORIZON_BUCKETS) -> list[str]:
    """Bucket labels in horizon order."""
    return [f"{lo}-{hi}" for lo, hi in sorted(buckets)]


def increments_from_residuals(
    residuals: dict[str, np.ndarray],
    q: float,
    buckets=HORIZON_BUCKETS,
    monotone: bool = True,
    floor_at_forecast: bool = True,
) -> dict[str, float]:
    """Band increment per bucket from residual samples.

    `monotone` enforces non-decreasing width across horizon buckets by taking
    a running maximum in horizon order. The empirical quantiles here *fall*
    with horizon, because the selected model increasingly over-forecasts
    during the post-bump slide (DECISIONS entry 22) - so the raw numbers would
    promise that a two-hour-ahead forecast needs less headroom than a
    thirty-minute one. That is an artefact of one episode, not a property of
    forecast uncertainty, which cannot shrink as the horizon grows. The
    constraint encodes that prior and is deliberately conservative.
    """
    raw: dict[str, float] = {}
    for label in bucket_labels(buckets):
        sample = residuals.get(label)
        if sample is None or len(sample) == 0:
            continue
        value = float(np.quantile(sample, q))
        raw[label] = max(0.0, value) if floor_at_forecast else value

    if not monotone:
        return raw

    out: dict[str, float] = {}
    running = -np.inf
    for label in bucket_labels(buckets):
        if label not in raw:
            continue
        running = max(running, raw[label])
        out[label] = running
    return out


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
    monotone: bool = True,
) -> pd.DataFrame:
    """Replay a band policy over the folds and measure what it would cost.

    For each fold the band is built from the *other* folds' residuals when
    `leave_one_fold_out` is set, so a fold is never provisioned using its own
    errors, and the monotone constraint is applied within each fold's band so
    the replayed policy is exactly the one that would ship.

    Reported per bucket:

    - `breach_rate`    share of points where the actual exceeded the band
      (compare against 1 - q)
    - `mean_idle`      average unused headroom, in CPU points
    - `mean_increment` average band width added above the point forecast

    **Caveat.** Leave-one-fold-out removes a fold's own residuals but not the
    overlap between folds: expanding-window origins share all earlier
    observations, so the held-out fold's errors remain correlated with the
    ones used to build its band. These breach rates are therefore still
    optimistic, just less so than in-sample ones. They are not out-of-sample
    estimates in the independent-sample sense and carry no confidence
    interval.
    """
    frame = label_buckets(preds[preds["model"] == model], buckets)
    folds = sorted(frame["fold"].unique())
    labels = bucket_labels(buckets)
    rows = []

    for fold in folds:
        held = frame[frame["fold"] == fold]
        source = frame[frame["fold"] != fold] if leave_one_fold_out else frame
        if held.empty or source.empty:
            continue

        residuals = {
            label: group["under_forecast"].to_numpy(dtype=float)
            for label, group in source.groupby("bucket", sort=False)
        }
        increments = increments_from_residuals(
            residuals, q, buckets=buckets, monotone=monotone, floor_at_forecast=floor_at_forecast
        )

        for label in labels:
            bucket_points = held[held["bucket"] == label]
            if bucket_points.empty or label not in increments:
                continue
            increment = increments[label]
            band = bucket_points["y_pred"] + increment
            breach = bucket_points["y_true"] > band
            slack = (band - bucket_points["y_true"]).clip(lower=0.0)
            rows.append(
                {
                    "bucket": label,
                    "fold": fold,
                    "n": len(bucket_points),
                    "increment": increment,
                    "n_breach": int(breach.sum()),
                    "sum_idle": float(slack.sum()),
                    "max_shortfall": float((bucket_points["y_true"] - band).max()),
                }
            )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail

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
    summary = summary.reindex([b for b in labels if b in summary.index])
    return summary.reset_index()


def overall_breach_rate(
    preds: pd.DataFrame, model: str, q: float, buckets=HORIZON_BUCKETS, **kwargs
) -> float:
    """Horizon-weighted leave-one-fold-out breach rate at nominal `q`."""
    replayed = replay_policy(preds, model, q, buckets=buckets, **kwargs)
    if replayed.empty:
        return float("nan")
    return float(np.average(replayed["breach_rate"], weights=replayed["n"]))


def calibrate_nominal_q(
    preds: pd.DataFrame,
    model: str,
    target_breach: float,
    buckets=HORIZON_BUCKETS,
    grid=None,
    monotone: bool = True,
) -> pd.DataFrame:
    """Sweep nominal q and report the realised leave-one-fold-out breach rate.

    The nominal quantile and the achieved breach rate are not the same number:
    a band built at q=0.90 breached 12.5% of the time out of fold, not 10%.
    Shipping the nominal q that *achieves* the policy's target breach rate
    means the cost assumption (`q = C_under / (C_under + C_over)`) is honoured
    in practice rather than only in the formula.
    """
    if not 0 < target_breach < 1:
        raise ValueError("target_breach must be in (0, 1)")
    candidates = np.arange(0.80, 0.996, 0.005) if grid is None else np.asarray(grid, dtype=float)

    rows = []
    for q in candidates:
        replayed = replay_policy(
            preds, model, float(q), buckets=buckets, monotone=monotone
        )
        if replayed.empty:
            continue
        rows.append(
            {
                "nominal_q": round(float(q), 4),
                "lofo_breach_rate": float(
                    np.average(replayed["breach_rate"], weights=replayed["n"])
                ),
                "mean_increment": float(
                    np.average(replayed["mean_increment"], weights=replayed["n"])
                ),
                "mean_idle": float(np.average(replayed["mean_idle"], weights=replayed["n"])),
                "worst_shortfall": float(replayed["worst_shortfall"].max()),
            }
        )
    table = pd.DataFrame(rows)
    table["meets_target"] = table["lofo_breach_rate"] <= target_breach
    table["target_breach"] = target_breach
    return table


def select_nominal_q(
    preds: pd.DataFrame,
    model: str,
    target_breach: float,
    buckets=HORIZON_BUCKETS,
    grid=None,
    monotone: bool = True,
) -> float:
    """Smallest nominal q whose leave-one-fold-out breach rate meets the target.

    Smallest, because every increase in q buys compliance with idle capacity;
    the cheapest band that honours the stated cost ratio is the right one.
    """
    table = calibrate_nominal_q(
        preds, model, target_breach, buckets=buckets, grid=grid, monotone=monotone
    )
    meeting = table[table["meets_target"]]
    if meeting.empty:
        raise ValueError(
            f"no nominal q in the grid reaches a breach rate of {target_breach}; "
            "the residual sample is too small or too skewed"
        )
    return float(meeting["nominal_q"].iloc[0])


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
    preds: pd.DataFrame,
    model: str,
    q: float,
    buckets=HORIZON_BUCKETS,
    monotone: bool = True,
) -> dict[str, float]:
    """Band increments for the real forecast, from all training folds.

    Used once, at prediction time. Pooling every fold is correct here because
    all of them are training-window data; the leave-one-fold-out replay exists
    to estimate honestly what this policy *will* cost, not to build it. The
    same monotone constraint is applied, so the shipped band has the shape
    that was replayed.
    """
    frame = label_buckets(preds[preds["model"] == model], buckets)
    residuals = {
        str(label): group["under_forecast"].to_numpy(dtype=float)
        for label, group in frame.groupby("bucket", sort=False)
    }
    return increments_from_residuals(residuals, q, buckets=buckets, monotone=monotone)
