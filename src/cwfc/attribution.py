from __future__ import annotations

import numpy as np
import pandas as pd


def vm_timestamp_matrix(
    readings: pd.DataFrame,
    timestamp_col: str = "timestamp",
    vm_col: str = "vm_id",
    value_col: str = "avg_cpu",
) -> pd.DataFrame:
    """Wide VM x timestamp matrix of CPU values (rows = VMs, cols sorted)."""
    wide = readings.pivot_table(
        index=vm_col, columns=timestamp_col, values=value_col, aggfunc="mean"
    )
    return wide.reindex(sorted(wide.columns), axis=1)


def cycle_jump_contributions(
    matrix: pd.DataFrame, period: int, dip_phase: int, rebound_phase: int
) -> pd.Series:
    """Per-VM mean change from the dip phase to the rebound phase.

    For every cycle in which both phases are observed, take each VM's value at
    the rebound position minus its value at the dip position, then average
    those jumps over cycles. Because the aggregate target is an unweighted mean
    over VMs, the aggregate jump is exactly the mean of these per-VM jumps —
    so each VM's share of the total is directly interpretable.
    """
    n = matrix.shape[1]
    dips = [i for i in range(n) if i % period == dip_phase]
    pairs = [
        (d, d + (rebound_phase - dip_phase) % period)
        for d in dips
        if d + (rebound_phase - dip_phase) % period < n
    ]
    if not pairs:
        raise ValueError("no complete dip/rebound pair inside the window")

    values = matrix.to_numpy(dtype=float)
    jumps = np.stack([values[:, r] - values[:, d] for d, r in pairs], axis=1)
    return pd.Series(np.nanmean(jumps, axis=1), index=matrix.index, name="jump")


def window_shift_contributions(
    matrix: pd.DataFrame, window: slice, baseline: slice
) -> pd.Series:
    """Per-VM mean CPU inside `window` minus mean CPU inside `baseline`."""
    values = matrix.to_numpy(dtype=float)
    shift = np.nanmean(values[:, window], axis=1) - np.nanmean(values[:, baseline], axis=1)
    return pd.Series(shift, index=matrix.index, name="shift")


def movement_summary(contributions: pd.Series) -> dict:
    """Net versus gross decomposition of a set of signed contributions.

    The aggregate series only shows `net`. `gross` is the total absolute
    movement underneath it, and `up`/`down` are the two sides. A small net on
    a large gross means the aggregate is a residual of offsetting VM-level
    movement, which is a materially different operational story from every VM
    drifting the same way.
    """
    values = contributions.to_numpy(dtype=float)
    up = float(values[values > 0].sum())
    down = float(values[values < 0].sum())
    return {
        "net": float(values.sum()),
        "gross": float(np.abs(values).sum()),
        "up": up,
        "down": down,
        "cancellation_ratio": float(np.abs(values).sum() / abs(values.sum()))
        if values.sum() != 0
        else np.nan,
        "n_up": int((values > 0).sum()),
        "n_down": int((values < 0).sum()),
    }


def contribution_shares(
    contributions: pd.Series, meta: pd.DataFrame, by: str, vm_col: str = "vm_id"
) -> pd.DataFrame:
    """Net and gross share of an aggregate move, per level of `by`.

    - `share_of_net`   group's summed signed contribution / overall net move
    - `share_of_gross` group's absolute movement / overall absolute movement
    - `lift`           share_of_net / vm_share: 1.0 means the group moved in
      proportion to its size, >1 means it drove the net move
    - `lift_gross`     same against gross, i.e. "was this group unusually
      *active*", regardless of direction

    Net shares can exceed 100% or go negative whenever groups move in
    opposite directions, so they are always reported next to the gross view.
    """
    joined = contributions.rename("contribution").to_frame().join(
        meta.set_index(vm_col)[[by]], how="left"
    )
    joined["abs_contribution"] = joined["contribution"].abs()
    net_total = float(joined["contribution"].sum())
    gross_total = float(joined["abs_contribution"].sum())
    n_vms = len(joined)

    grouped = joined.groupby(by, dropna=False)
    out = pd.DataFrame(
        {
            "n_vms": grouped.size(),
            "net": grouped["contribution"].sum(),
            "gross": grouped["abs_contribution"].sum(),
            "up": grouped["contribution"].apply(lambda s: float(s[s > 0].sum())),
            "down": grouped["contribution"].apply(lambda s: float(s[s < 0].sum())),
            "mean_contribution": grouped["contribution"].mean(),
        }
    )
    out["vm_share"] = out["n_vms"] / n_vms
    out["share_of_net"] = out["net"] / net_total if net_total != 0 else np.nan
    out["share_of_gross"] = out["gross"] / gross_total if gross_total != 0 else np.nan
    out["lift"] = out["share_of_net"] / out["vm_share"]
    out["lift_gross"] = out["share_of_gross"] / out["vm_share"]
    return out.sort_values("share_of_net", ascending=False)


def top_vm_shares(
    contributions: pd.Series, fractions=(0.001, 0.01, 0.05, 0.10, 0.25)
) -> pd.DataFrame:
    """Net and gross share of the move explained by the top-contributing VMs.

    VMs are ranked by signed contribution for the net column and by absolute
    contribution for the gross column, so the two columns answer different
    questions: "how much of the net move do the biggest movers explain" versus
    "how much of all the movement do the most active VMs account for".
    """
    net_total = float(contributions.sum())
    gross_total = float(contributions.abs().sum())
    by_net = contributions.sort_values(ascending=False)
    by_gross = contributions.abs().sort_values(ascending=False)

    rows = []
    for f in fractions:
        k = max(1, int(round(f * len(contributions))))
        rows.append(
            {
                "top_fraction": f,
                "n_vms": k,
                "share_of_net": float(by_net.iloc[:k].sum() / net_total)
                if net_total
                else np.nan,
                "share_of_gross": float(by_gross.iloc[:k].sum() / gross_total)
                if gross_total
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
