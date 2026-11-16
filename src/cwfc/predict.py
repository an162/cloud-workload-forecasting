"""Generate predictions.csv for the holdout horizon.

Reads the training window only. The holdout rows contribute nothing but their
`timestamp` values, which the submission must reproduce exactly; no holdout
target value is read, so corrupting them cannot change the output (see
tests/test_predict.py::test_no_lookahead).

Scoring lives in a separate command on purpose, so that producing a
submission can never be confused with evaluating one.

    python -m cwfc.predict --data challenge_data --output predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc import frozen
from cwfc.data import load_cpu_readings
from cwfc.evaluate import rolling_origin_predictions
from cwfc.policy import (
    cost_ratio_for_q,
    final_band_increments,
    label_buckets,
    overall_breach_rate,
    select_nominal_q,
)
from cwfc.seasonal import detect_period
from cwfc.targets import build_targets, chronological_split, diagnose_targets

TARGETS = (frozen.AGGREGATE_CPU, frozen.ACTIVE_VM_COUNT)
TIMESTAMP = "timestamp"


def _bound(values: np.ndarray, target: str) -> np.ndarray:
    lower, upper = frozen.OUTPUT_BOUNDS[target]
    out = np.asarray(values, dtype=float)
    if lower is not None:
        out = np.maximum(out, lower)
    if upper is not None:
        out = np.minimum(out, upper)
    if target in frozen.INTEGER_TARGETS:
        out = np.rint(out)
    return out


def forecast_targets(
    readings: pd.DataFrame, holdout_size: int = frozen.HOLDOUT_SIZE
) -> pd.DataFrame:
    """Fit each target's frozen model on the training window and forecast."""
    targets = build_targets(readings)
    train, holdout = chronological_split(targets, holdout_size=holdout_size)

    # Timestamps only: the holdout's target columns are never touched.
    horizon_timestamps = holdout[TIMESTAMP].to_numpy()

    out: dict[str, np.ndarray] = {TIMESTAMP: horizon_timestamps}
    for target in TARGETS:
        history = train[target].to_numpy(dtype=float)
        if np.isnan(history).any():
            raise ValueError(
                f"training window for {target!r} contains gaps; targets are never "
                "filled, so resolve them explicitly before forecasting"
            )
        model = frozen.MODEL_BY_TARGET[target]
        period = detect_period(history)
        values = _bound(model(history, holdout_size, period), target)
        out[target] = values.astype("int64") if target in frozen.INTEGER_TARGETS else values

    # Column order is part of the submission contract, not an implementation
    # detail, so it comes from the frozen spec rather than insertion order.
    return pd.DataFrame(out)[list(frozen.SUBMISSION_COLUMNS)]


def capacity_band(
    readings: pd.DataFrame,
    predictions: pd.DataFrame,
    holdout_size: int = frozen.HOLDOUT_SIZE,
    target_breach: float = frozen.TARGET_BREACH_RATE,
    nominal_q: float | None = None,
) -> pd.DataFrame:
    """Upper capacity band for the CPU forecast, from training folds only.

    The nominal quantile is calibrated by leave-one-fold-out replay so the
    band achieves `target_breach`, rather than assumed to achieve it. Pass
    `nominal_q` to override the calibration explicitly.
    """
    targets = build_targets(readings)
    train, _ = chronological_split(targets, holdout_size=holdout_size)
    history = train[frozen.AGGREGATE_CPU].to_numpy(dtype=float)

    folds = rolling_origin_predictions(
        history,
        horizon=holdout_size,
        step=frozen.ORIGIN_STEP,
        models={frozen.SELECTED_MODEL_NAME: frozen.SELECTED_MODEL},
    )
    q = (
        select_nominal_q(folds, frozen.SELECTED_MODEL_NAME, target_breach)
        if nominal_q is None
        else float(nominal_q)
    )
    increments = final_band_increments(
        folds, frozen.SELECTED_MODEL_NAME, q, monotone=frozen.BAND_MONOTONE
    )
    achieved = overall_breach_rate(folds, frozen.SELECTED_MODEL_NAME, q)

    steps = pd.DataFrame({"step": np.arange(1, holdout_size + 1)})
    steps["model"] = frozen.SELECTED_MODEL_NAME
    steps["error"] = 0.0
    buckets = label_buckets(steps)[["step", "bucket"]]

    band = predictions[[TIMESTAMP, frozen.AGGREGATE_CPU]].copy()
    band["step"] = np.arange(1, len(band) + 1)
    band = band.merge(buckets, on="step", how="left")
    band["band_increment"] = band["bucket"].map(increments).astype(float)
    band["upper_band"] = np.minimum(
        band[frozen.AGGREGATE_CPU] + band["band_increment"], frozen.OUTPUT_BOUNDS[frozen.AGGREGATE_CPU][1]
    )
    band["nominal_q"] = q
    band["target_breach_rate"] = target_breach
    band["lofo_breach_rate"] = achieved
    return band[
        [
            TIMESTAMP,
            "step",
            "bucket",
            frozen.AGGREGATE_CPU,
            "band_increment",
            "upper_band",
            "nominal_q",
            "target_breach_rate",
            "lofo_breach_rate",
        ]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="challenge_data", help="directory holding the release")
    parser.add_argument("--output", default="predictions.csv", help="prediction file to write")
    parser.add_argument(
        "--holdout", type=int, default=frozen.HOLDOUT_SIZE, help="number of trailing timestamps to forecast"
    )
    parser.add_argument("--band-output", default=None, help="optional capacity-band CSV")
    parser.add_argument(
        "--target-breach",
        type=float,
        default=frozen.TARGET_BREACH_RATE,
        help="breach rate the band must achieve; encodes the under/over cost ratio",
    )
    parser.add_argument(
        "--band-q", type=float, default=None, help="override the calibrated nominal quantile"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    readings = load_cpu_readings(args.data)
    report = diagnose_targets(readings)
    predictions = forecast_targets(readings, holdout_size=args.holdout)

    output = Path(args.output)
    predictions.to_csv(output, index=False)

    if not args.quiet:
        print(f"data            : {args.data}")
        print(f"rows read       : {report.n_rows}  invalid={report.n_invalid} "
              f"out_of_range={report.n_out_of_range} duplicate_rows={report.n_duplicate_rows}")
        print(f"grid            : {report.n_grid_slots} slots, step={report.step}, "
              f"absent={report.n_slots_absent}")
        print(f"models          : " + ", ".join(
            f"{t}={frozen.MODEL_NAME_BY_TARGET[t]}" for t in TARGETS))
        print(f"holdout horizon : {args.holdout} timestamps "
              f"[{predictions[TIMESTAMP].iloc[0]}..{predictions[TIMESTAMP].iloc[-1]}]")
        print(f"wrote           : {output} ({len(predictions)} rows)")

    if args.band_output:
        band = capacity_band(
            readings,
            predictions,
            holdout_size=args.holdout,
            target_breach=args.target_breach,
            nominal_q=args.band_q,
        )
        band.to_csv(args.band_output, index=False)
        if not args.quiet:
            q = float(band["nominal_q"].iloc[0])
            print(f"band            : target breach {args.target_breach:.0%} "
                  f"(cost ratio {cost_ratio_for_q(1 - args.target_breach):.0f}x), "
                  f"calibrated nominal q={q:.3f}, "
                  f"replayed breach {float(band['lofo_breach_rate'].iloc[0]):.1%}")
            print(f"wrote           : {args.band_output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
