from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def group_series(
    readings: pd.DataFrame,
    meta: pd.DataFrame,
    by: str,
    timestamp_col: str = "timestamp",
    vm_col: str = "vm_id",
    value_col: str = "avg_cpu",
) -> pd.DataFrame:
    """Mean CPU per group per timestamp: index = timestamp, columns = group."""
    joined = readings.merge(meta[[vm_col, by]], on=vm_col, how="left")
    wide = joined.pivot_table(
        index=timestamp_col, columns=by, values=value_col, aggfunc="mean"
    )
    return wide.sort_index()


def trailing_growth(series: pd.DataFrame, window: int) -> pd.DataFrame:
    """Total least-squares growth across the trailing `window` observations.

    For each column and each time t, fit a straight line to the last `window`
    points ending at t and report the rise across that window
    (`slope * (window - 1)`), in the units of the series.

    Because the design matrix is fixed, the fit reduces to one linear filter,
    which is applied as a single strided dot product rather than a Python loop.
    Rows before the first complete window are NaN, so the detector is never
    evaluated on a partial window.

    Only observations up to and including t enter the value at t: this is a
    causal statistic, computable at time t in production.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    values = series.to_numpy(dtype=float)
    n_rows = values.shape[0]
    out = np.full(values.shape, np.nan)
    if n_rows < window:
        return pd.DataFrame(out, index=series.index, columns=series.columns)

    positions = np.arange(window, dtype=float)
    centred = positions - positions.mean()
    weights = centred / float((centred**2).sum())

    windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
    slopes = np.einsum("tgw,w->tg", windows, weights)
    out[window - 1 :, :] = slopes * (window - 1)
    return pd.DataFrame(out, index=series.index, columns=series.columns)


def _episode_starts(flags: np.ndarray) -> np.ndarray:
    """Rising edges per column: a new alarm episode, not a continuing one."""
    previous = np.vstack([np.zeros((1, flags.shape[1]), dtype=bool), flags[:-1]])
    return flags & ~previous


@dataclass(frozen=True)
class DetectorResult:
    threshold: float
    n_eligible_steps: int
    hours_monitored: float
    n_drivers: int
    n_detected_before_event: int
    recall: float
    precision: float
    n_groups_flagged_before_event: int
    median_lead_steps: float
    max_lead_steps: float
    median_lead_minutes: float
    max_lead_minutes: float
    false_alarm_episodes: int
    false_alarms_per_hour: float
    total_episodes: int
    flag_rate: float
    lead_by_driver: dict


def evaluate_detector(
    growth: pd.DataFrame,
    drivers: list,
    event_index: int,
    threshold: float,
    step_seconds: int,
) -> DetectorResult:
    """Score a prospective growth detector against a retrospective label.

    At every step the detector flags each group whose trailing growth is at or
    above `threshold`, using only data up to that step. Groups in `drivers`
    are the ones that later turned out to drive the event at `event_index`;
    that label is retrospective and is used for *scoring only*, never by the
    detector.

    - lead time: steps between a driver's first pre-event flag and the event
    - precision: of the groups flagged before the event, the share that are
      drivers (a narrow label, so a non-driver flag is not necessarily a
      spurious signal, just one that did not move the aggregate)
    - false alarms per hour: non-driver alarm *episodes* (rising edges, so a
      persistent flag counts once) per monitored hour over the whole window
    """
    values = growth.to_numpy(dtype=float)
    eligible = ~np.isnan(values).all(axis=1)
    flags = np.where(np.isnan(values), False, values >= threshold)

    columns = list(growth.columns)
    driver_set = {d for d in drivers if d in growth.columns}
    driver_cols = [columns.index(d) for d in driver_set]

    pre = flags[:event_index, :]
    flagged_pre = pre.any(axis=0)
    flagged_groups = {columns[i] for i in np.flatnonzero(flagged_pre)}

    leads: dict = {}
    for col_index in driver_cols:
        hits = np.flatnonzero(pre[:, col_index])
        if hits.size:
            leads[columns[col_index]] = int(event_index - hits[0])

    lead_values = np.array(sorted(leads.values()), dtype=float)
    episode_starts = _episode_starts(flags)
    non_driver_cols = [i for i in range(len(columns)) if columns[i] not in driver_set]
    false_episodes = int(episode_starts[:, non_driver_cols].sum())

    n_eligible = int(eligible.sum())
    hours = n_eligible * step_seconds / 3600.0

    return DetectorResult(
        threshold=float(threshold),
        n_eligible_steps=n_eligible,
        hours_monitored=hours,
        n_drivers=len(driver_set),
        n_detected_before_event=len(leads),
        recall=len(leads) / len(driver_set) if driver_set else float("nan"),
        precision=(
            len(flagged_groups & driver_set) / len(flagged_groups)
            if flagged_groups
            else float("nan")
        ),
        n_groups_flagged_before_event=len(flagged_groups),
        median_lead_steps=float(np.median(lead_values)) if lead_values.size else float("nan"),
        max_lead_steps=float(lead_values.max()) if lead_values.size else float("nan"),
        median_lead_minutes=(
            float(np.median(lead_values) * step_seconds / 60.0)
            if lead_values.size
            else float("nan")
        ),
        max_lead_minutes=(
            float(lead_values.max() * step_seconds / 60.0) if lead_values.size else float("nan")
        ),
        false_alarm_episodes=false_episodes,
        false_alarms_per_hour=false_episodes / hours if hours > 0 else float("nan"),
        total_episodes=int(episode_starts.sum()),
        flag_rate=float(flags[eligible].mean()) if n_eligible else float("nan"),
        lead_by_driver=dict(sorted(leads.items(), key=lambda kv: -kv[1])),
    )


def detector_sweep(
    growth: pd.DataFrame,
    drivers: list,
    event_index: int,
    thresholds,
    step_seconds: int,
) -> pd.DataFrame:
    """Run `evaluate_detector` across thresholds and tabulate the trade-off."""
    rows = []
    for threshold in thresholds:
        r = evaluate_detector(growth, drivers, event_index, threshold, step_seconds)
        rows.append(
            {
                "threshold": r.threshold,
                "recall": r.recall,
                "detected": f"{r.n_detected_before_event}/{r.n_drivers}",
                "precision": r.precision,
                "n_flagged_pre": r.n_groups_flagged_before_event,
                "median_lead_min": r.median_lead_minutes,
                "max_lead_min": r.max_lead_minutes,
                "false_alarms_per_hour": r.false_alarms_per_hour,
                "false_episodes": r.false_alarm_episodes,
                "total_episodes": r.total_episodes,
                "flag_rate": r.flag_rate,
            }
        )
    return pd.DataFrame(rows)
