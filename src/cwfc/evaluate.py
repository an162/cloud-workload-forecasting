from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from cwfc import baselines, metrics
from cwfc.seasonal import detect_period, min_length_for_period
from cwfc.validation import expanding_window_folds

ModelFn = Callable[[np.ndarray, int, int | None], np.ndarray]

BASELINES: dict[str, ModelFn] = {
    "persistence": baselines.persistence,
    "last_cycle_mean": baselines.last_cycle_mean,
    "seasonal_naive": baselines.seasonal_naive,
    "train_mean": baselines.train_mean,
}

PEAK_QUANTILE = 0.9
HORIZON_BUCKETS = ((1, 6), (7, 12), (13, 24))


def fold_plan(n_obs: int, horizon: int, period: int | None, step: int = 3) -> list:
    """Folds sized by PLAN 2: min_train = max(2H, min_length_for_period(P)).

    The minimum training length is the larger of two horizons (so every fold
    has at least as much history as it must forecast) and the shortest window
    on which `detect_period` can actually test the 2P harmonic (so no fold is
    handed a seasonality question its detector cannot answer, and every fold
    makes the same seasonal decision for the same reason). An origin step above
    1 spreads the origins across phases of the cycle rather than over-sampling
    one phase.
    """
    min_train = (
        2 * horizon
        if period is None
        else max(2 * horizon, min_length_for_period(int(period)))
    )
    return expanding_window_folds(n_obs, horizon=horizon, min_train=min_train, step=step)


def rolling_origin_predictions(
    series,
    horizon: int,
    step: int = 3,
    models: dict[str, ModelFn] | None = None,
    detect_seasonality: bool = True,
) -> pd.DataFrame:
    """Run every model over expanding-window folds and return tidy predictions.

    One row per (model, fold, horizon step). Everything a metric needs later
    is attached at the row level and computed from that fold's training slice
    only: the detected period, the MASE scale, and the peak threshold. Nothing
    is derived from a fold's test window, so no fold can leak into its own
    score.
    """
    values = np.asarray(series, dtype=float)
    if np.isnan(values).any():
        raise ValueError("series must not contain NaN; resolve gaps explicitly first")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    active = BASELINES if models is None else models
    global_period = detect_period(values) if detect_seasonality else None
    folds = fold_plan(len(values), horizon=horizon, period=global_period, step=step)
    if not folds:
        raise ValueError(
            f"no fold fits: {len(values)} observations is too short for horizon {horizon}"
        )

    rows: list[dict] = []
    for fold in folds:
        train = values[fold.train_slice]
        test = values[fold.test_slice]
        # Per-fold, training-only quantities.
        period = detect_period(train) if detect_seasonality else None
        scale = metrics.naive_scale(train)
        peak_threshold = float(np.quantile(train, PEAK_QUANTILE))

        for name, fn in active.items():
            pred = np.asarray(fn(train, horizon, period), dtype=float)
            if pred.shape != test.shape:
                raise ValueError(f"model {name!r} returned {pred.shape}, expected {test.shape}")
            for h in range(horizon):
                rows.append(
                    {
                        "model": name,
                        "fold": fold.index,
                        "origin": fold.origin,
                        "step": h + 1,
                        "position": fold.origin + h,
                        "y_true": test[h],
                        "y_pred": pred[h],
                        "period": period,
                        "mase_scale": scale,
                        "peak_threshold": peak_threshold,
                    }
                )

    out = pd.DataFrame(rows)
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["is_peak"] = out["y_true"] >= out["peak_threshold"]
    return out


def _metric_block(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "mae": metrics.mae(group["y_true"], group["y_pred"]),
            "rmse": metrics.rmse(group["y_true"], group["y_pred"]),
            "bias": metrics.bias(group["y_true"], group["y_pred"]),
            "wape": metrics.wape(group["y_true"], group["y_pred"]),
            "mase": metrics.mase(
                group["y_true"], group["y_pred"], float(group["mase_scale"].iloc[0])
            ),
        }
    )


def fold_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-model, per-fold metrics over that fold's whole horizon path."""
    return (
        preds.groupby(["model", "fold"], sort=False)[list(preds.columns)]
        .apply(_metric_block, include_groups=False)
        .reset_index()
    )


def model_summary(
    preds: pd.DataFrame, baseline_names: tuple[str, ...] = ("persistence", "last_cycle_mean")
) -> pd.DataFrame:
    """Mean and spread of each metric across folds, plus skill vs baselines.

    Folds overlap, so their errors are correlated: the standard deviation is
    reported as a spread indicator only and no significance test is implied
    (PLAN 2).
    """
    per_fold = fold_metrics(preds)
    summary = per_fold.groupby("model", sort=False).agg(
        mae=("mae", "mean"),
        mae_sd=("mae", "std"),
        rmse=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        bias=("bias", "mean"),
        wape=("wape", "mean"),
        mase=("mase", "mean"),
        n_folds=("fold", "nunique"),
    )
    for base in baseline_names:
        if base in summary.index:
            base_mae = summary.loc[base, "mae"]
            summary[f"skill_vs_{base}"] = (
                np.nan if base_mae == 0 else 1.0 - summary["mae"] / base_mae
            )
    return summary.sort_values("mae")


def horizon_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Metrics by horizon step, averaged over folds."""
    return (
        preds.groupby(["model", "step"], sort=False)
        .agg(
            mae=("abs_error", "mean"),
            rmse=("error", lambda e: float(np.sqrt(np.mean(np.square(e))))),
            bias=("error", "mean"),
        )
        .reset_index()
    )


def horizon_bucket_errors(
    preds: pd.DataFrame, buckets: tuple[tuple[int, int], ...] = HORIZON_BUCKETS
) -> pd.DataFrame:
    """Pooled residuals grouped into horizon buckets, for later interval work."""
    labelled = preds.copy()
    labels = pd.Series(pd.NA, index=labelled.index, dtype="object")
    for lo, hi in buckets:
        labels = labels.mask(labelled["step"].between(lo, hi), f"{lo}-{hi}")
    labelled["bucket"] = labels
    return (
        labelled.dropna(subset=["bucket"])
        .groupby(["model", "bucket"], sort=False)
        .agg(
            n=("error", "size"),
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
            sd=("error", "std"),
            q80_under=("error", lambda e: float(np.quantile(-e, 0.80))),
            q90_under=("error", lambda e: float(np.quantile(-e, 0.90))),
            q95_under=("error", lambda e: float(np.quantile(-e, 0.95))),
        )
        .reset_index()
    )


def predictability_curve(preds: pd.DataFrame, reference: str = "train_mean") -> pd.DataFrame:
    """Skill vs the reference model at each horizon step.

    The step at which skill decays to ~0 is the predictability horizon: beyond
    it, knowing the recent past buys nothing over knowing the long-run level.
    """
    by_step = horizon_metrics(preds)
    ref = by_step.loc[by_step["model"] == reference, ["step", "mae"]].rename(
        columns={"mae": "ref_mae"}
    )
    merged = by_step.merge(ref, on="step", how="left")
    merged[f"skill_vs_{reference}"] = np.where(
        merged["ref_mae"] > 0, 1.0 - merged["mae"] / merged["ref_mae"], np.nan
    )
    return merged


def peak_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Errors restricted to peak actuals (>= the fold's training 90th pct)."""
    peaks = preds[preds["is_peak"]]
    if peaks.empty:
        return pd.DataFrame()
    return (
        peaks.groupby("model", sort=False)
        .agg(
            n_peak=("abs_error", "size"),
            peak_mae=("abs_error", "mean"),
            peak_bias=("error", "mean"),
            peak_rmse=("error", lambda e: float(np.sqrt(np.mean(np.square(e))))),
        )
        .sort_values("peak_mae")
    )


def failure_analysis(preds: pd.DataFrame, model: str) -> dict:
    """Worst fold and worst single step for one model, with context."""
    subset = preds[preds["model"] == model]
    per_fold = (
        subset.groupby(["fold", "origin"], sort=False)["abs_error"].mean().sort_values()
    )
    worst_row = subset.loc[subset["abs_error"].idxmax()]
    worst_fold, worst_origin = per_fold.index[-1]
    return {
        "model": model,
        "worst_fold": int(worst_fold),
        "worst_fold_origin": int(worst_origin),
        "worst_fold_mae": float(per_fold.iloc[-1]),
        "best_fold": int(per_fold.index[0][0]),
        "best_fold_mae": float(per_fold.iloc[0]),
        "worst_step_position": int(worst_row["position"]),
        "worst_step_h": int(worst_row["step"]),
        "worst_step_abs_error": float(worst_row["abs_error"]),
        "worst_step_y_true": float(worst_row["y_true"]),
        "worst_step_y_pred": float(worst_row["y_pred"]),
    }


# ---------------------------------------------------------------------------
# Holdout scoring CLI
#
# Deliberately separate from `cwfc.predict` (PLAN 7): producing a submission
# must never print a holdout score, or every development rerun would leak it
# into the author's decisions. This module reads holdout target values; the
# prediction path does not.
# ---------------------------------------------------------------------------


def holdout_predictions(
    readings: "pd.DataFrame",
    holdout_size: int,
    targets: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Score-ready frame: every declared model forecast against the holdout.

    One "fold" whose training slice is the whole training window. Each model
    is refitted on all of it, as PLAN 2 prescribes, and the MASE scale and
    peak threshold come from that same training window - never from the
    holdout being scored.
    """
    from cwfc import frozen
    from cwfc.targets import build_targets, chronological_split

    model_set = dict(BASELINES)
    model_set[frozen.SELECTED_MODEL_NAME] = frozen.SELECTED_MODEL

    target_names = targets or tuple(frozen.MODEL_BY_TARGET)
    built = build_targets(readings)
    train, holdout = chronological_split(built, holdout_size=holdout_size)

    rows: list[dict] = []
    for target in target_names:
        history = train[target].to_numpy(dtype=float)
        actual = holdout[target].to_numpy(dtype=float)
        if np.isnan(history).any():
            raise ValueError(f"training window for {target!r} contains gaps")

        period = detect_period(history)
        scale = metrics.naive_scale(history)
        peak_threshold = float(np.quantile(history, PEAK_QUANTILE))

        for name, fn in model_set.items():
            pred = np.asarray(fn(history, holdout_size, period), dtype=float)
            for h in range(holdout_size):
                rows.append(
                    {
                        "target": target,
                        "model": name,
                        "fold": 0,
                        "origin": len(history),
                        "step": h + 1,
                        "position": len(history) + h,
                        "timestamp": int(holdout["timestamp"].iloc[h]),
                        "y_true": actual[h],
                        "y_pred": pred[h],
                        "period": period,
                        "mase_scale": scale,
                        "peak_threshold": peak_threshold,
                        "is_selected": name == frozen.MODEL_NAME_BY_TARGET.get(target),
                    }
                )

    out = pd.DataFrame(rows)
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["is_peak"] = out["y_true"] >= out["peak_threshold"]
    return out


def holdout_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Section 4 metric table for one target's holdout scores."""
    rows = []
    for model, group in scored.groupby("model", sort=False):
        rows.append(
            {
                "model": model,
                "mae": metrics.mae(group["y_true"], group["y_pred"]),
                "rmse": metrics.rmse(group["y_true"], group["y_pred"]),
                "bias": metrics.bias(group["y_true"], group["y_pred"]),
                "wape": metrics.wape(group["y_true"], group["y_pred"]),
                "mase": metrics.mase(
                    group["y_true"], group["y_pred"], float(group["mase_scale"].iloc[0])
                ),
                "selected": bool(group["is_selected"].iloc[0]),
            }
        )
    table = pd.DataFrame(rows).set_index("model")

    for base in ("persistence", "last_cycle_mean", "train_mean"):
        if base in table.index:
            base_mae = table.loc[base, "mae"]
            table[f"skill_vs_{base}"] = (
                np.nan if base_mae == 0 else 1.0 - table["mae"] / base_mae
            )
    return table.sort_values("mae")
