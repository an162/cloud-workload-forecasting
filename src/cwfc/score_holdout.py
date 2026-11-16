"""Score the holdout. One shot, after the model and policy were frozen.

    python -m cwfc.score_holdout --data challenge_data --predictions predictions.csv

This is the only entry point that reads holdout target values. It changes
nothing: by the time it runs, the model, the band, and the policy are fixed in
`cwfc.frozen` and tagged `frozen-before-holdout` in git.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc import frozen, metrics
from cwfc.data import load_cpu_readings
from cwfc.evaluate import (
    holdout_predictions,
    holdout_summary,
    peak_metrics,
    rolling_origin_predictions,
)
from cwfc.policy import final_band_increments, label_buckets, select_nominal_q
from cwfc.targets import AGGREGATE_CPU, build_targets, chronological_split


def band_report(readings: pd.DataFrame, scored: pd.DataFrame, holdout_size: int) -> dict:
    """Did the shipped capacity band hold on the holdout?"""
    train, _ = chronological_split(build_targets(readings), holdout_size=holdout_size)
    history = train[AGGREGATE_CPU].to_numpy(dtype=float)
    folds = rolling_origin_predictions(
        history,
        horizon=holdout_size,
        step=frozen.ORIGIN_STEP,
        models={frozen.SELECTED_MODEL_NAME: frozen.SELECTED_MODEL},
    )
    q = select_nominal_q(folds, frozen.SELECTED_MODEL_NAME, frozen.TARGET_BREACH_RATE)
    increments = final_band_increments(
        folds, frozen.SELECTED_MODEL_NAME, q, monotone=frozen.BAND_MONOTONE
    )

    selected = scored[
        (scored["target"] == AGGREGATE_CPU)
        & (scored["model"] == frozen.MODEL_NAME_BY_TARGET[AGGREGATE_CPU])
    ].copy()
    selected = label_buckets(selected)
    selected["increment"] = selected["bucket"].map(increments).astype(float)
    selected["upper_band"] = selected["y_pred"] + selected["increment"]
    selected["breach"] = selected["y_true"] > selected["upper_band"]
    selected["idle"] = (selected["upper_band"] - selected["y_true"]).clip(lower=0.0)

    per_bucket = selected.groupby("bucket", sort=False).agg(
        n=("breach", "size"),
        increment=("increment", "first"),
        breaches=("breach", "sum"),
        breach_rate=("breach", "mean"),
        mean_idle=("idle", "mean"),
        worst_shortfall=("error", lambda e: float((-e).max())),
    )
    return {
        "nominal_q": q,
        "target_breach": frozen.TARGET_BREACH_RATE,
        "overall_breach_rate": float(selected["breach"].mean()),
        "overall_mean_idle": float(selected["idle"].mean()),
        "n_breaches": int(selected["breach"].sum()),
        "n": len(selected),
        "per_bucket": per_bucket,
        "detail": selected,
    }


def check_submission(predictions_path: Path, scored: pd.DataFrame) -> dict:
    """Confirm the shipped file equals what the frozen pipeline produces."""
    shipped = pd.read_csv(predictions_path)
    rebuilt = {}
    for target, model in frozen.MODEL_NAME_BY_TARGET.items():
        rows = scored[(scored["target"] == target) & (scored["model"] == model)]
        rebuilt[target] = rows.sort_values("step")["y_pred"].to_numpy()

    checks = {
        "columns_match_spec": list(shipped.columns) == list(frozen.SUBMISSION_COLUMNS),
        "row_count": len(shipped) == len(next(iter(rebuilt.values()))),
        "timestamps_match": list(shipped["timestamp"])
        == list(scored.sort_values("step")["timestamp"].unique()),
    }
    for target, values in rebuilt.items():
        shipped_values = shipped[target].to_numpy(dtype=float)
        bounded = np.asarray(values, dtype=float)
        lower, upper = frozen.OUTPUT_BOUNDS[target]
        if lower is not None:
            bounded = np.maximum(bounded, lower)
        if upper is not None:
            bounded = np.minimum(bounded, upper)
        if target in frozen.INTEGER_TARGETS:
            bounded = np.rint(bounded)
        checks[f"{target}_matches_pipeline"] = bool(
            np.allclose(shipped_values, bounded, atol=1e-9)
        )
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--predictions", default="predictions.csv")
    parser.add_argument("--holdout", type=int, default=frozen.HOLDOUT_SIZE)
    parser.add_argument("--output", default=None, help="optional markdown report path")
    args = parser.parse_args(argv)

    readings = load_cpu_readings(args.data)
    scored = holdout_predictions(readings, holdout_size=args.holdout)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("# Holdout scores")
    emit()
    emit(f"Frozen model: `{frozen.SELECTED_MODEL_NAME}` | horizon: {args.holdout} timestamps")
    emit(f"Frozen at: {frozen.FROZEN_AT}")
    emit()

    submission = check_submission(Path(args.predictions), scored)
    emit("## Submission integrity")
    emit()
    for key, value in submission.items():
        emit(f"- {key}: **{'PASS' if value else 'FAIL'}**")
    emit()

    for target in frozen.MODEL_BY_TARGET:
        subset = scored[scored["target"] == target]
        actual = subset["y_true"].to_numpy()[: args.holdout]
        emit(f"## {target}")
        emit()
        emit(f"Actual over the holdout: mean={actual.mean():.4f} sd={actual.std(ddof=1):.4f} "
             f"min={actual.min():.4f} max={actual.max():.4f}")
        emit(f"Detected period on training: {subset['period'].iloc[0]} | "
             f"MASE scale: {subset['mase_scale'].iloc[0]:.4f} | "
             f"peak threshold (training p90): {subset['peak_threshold'].iloc[0]:.4f}")
        emit()
        table = holdout_summary(subset)
        emit("```")
        emit(table.round(4).to_string())
        emit("```")
        emit()

        peaks = peak_metrics(subset)
        if not peaks.empty:
            emit(f"Peak points (actual >= training p90): "
                 f"{int(subset.groupby('model')['is_peak'].sum().iloc[0])} of {args.holdout}")
            emit("```")
            emit(peaks.round(4).to_string())
            emit("```")
            emit()

        by_step = subset.pivot_table(index="step", columns="model", values="error")
        emit("Signed error by horizon step (positive = over-forecast):")
        emit("```")
        emit(by_step.round(3).to_string())
        emit("```")
        emit()

    emit("## Capacity band")
    emit()
    band = band_report(readings, scored, args.holdout)
    emit(f"Shipped nominal q={band['nominal_q']:.3f} for a target breach rate of "
         f"{band['target_breach']:.0%} (cost ratio {frozen.COST_RATIO:.0f}x)")
    emit(f"Holdout breaches: **{band['n_breaches']} of {band['n']} "
         f"({band['overall_breach_rate']:.1%})**, mean idle headroom "
         f"{band['overall_mean_idle']:.4f} points")
    emit()
    emit("```")
    emit(band["per_bucket"].round(4).to_string())
    emit("```")

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
