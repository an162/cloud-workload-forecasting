from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def infer_step(timestamps) -> int:
    """Infer the observation step as the modal diff between distinct timestamps."""
    uniq = pd.Index(pd.unique(pd.Series(timestamps).to_numpy())).sort_values()
    if len(uniq) < 2:
        raise ValueError("need at least two distinct timestamps to infer a step")
    diffs = pd.Series(uniq).diff().dropna().astype("int64")
    return int(diffs.mode().iat[0])


@dataclass(frozen=True)
class CadenceReport:
    """Description of the observation grid implied by a readings table."""

    n_timestamps: int
    first_timestamp: int
    last_timestamp: int
    expected_step: int
    step_counts: dict[int, int]
    n_gaps: int
    n_missing_slots: int
    gaps: list[tuple[int, int]]

    @property
    def is_regular(self) -> bool:
        return self.n_gaps == 0 and len(self.step_counts) <= 1


def cadence_report(frame: pd.DataFrame, timestamp_col: str = "timestamp") -> CadenceReport:
    """Summarise the cadence of the distinct timestamps in `frame`.

    The expected step is inferred as the most common difference between
    consecutive distinct timestamps, so nothing about the source cadence is
    assumed. Every distinct difference is reported alongside it, with counts,
    so the inference can be audited rather than trusted.

    A gap is any consecutive pair whose difference exceeds the expected step.
    `n_missing_slots` counts how many grid positions those gaps imply are absent.
    """
    if frame.empty:
        raise ValueError("frame must not be empty")

    grid = pd.Index(frame[timestamp_col].unique()).sort_values()
    if len(grid) < 2:
        raise ValueError("need at least two distinct timestamps to infer cadence")

    diffs = pd.Series(grid).diff().dropna().astype("int64")
    step_counts = {int(step): int(count) for step, count in diffs.value_counts().items()}
    expected_step = infer_step(grid)

    oversized = diffs[diffs > expected_step]
    gaps = [(int(grid[i - 1]), int(grid[i])) for i in oversized.index]
    n_missing_slots = int((oversized // expected_step - 1).sum())

    return CadenceReport(
        n_timestamps=len(grid),
        first_timestamp=int(grid[0]),
        last_timestamp=int(grid[-1]),
        expected_step=expected_step,
        step_counts=step_counts,
        n_gaps=len(gaps),
        n_missing_slots=n_missing_slots,
        gaps=gaps,
    )
