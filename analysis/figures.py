"""Figures for the report and the notebook.

One source of truth: the notebook imports these functions rather than
re-deriving anything, so a figure in the notebook and the same figure in
reports/figures/ cannot disagree.

Palette: the validated reference instance (categorical slots 1-3 blue/orange/
aqua, light surface #fcfcfb). Slots 1-3 clear every hard gate on the all-pairs
list; aqua sits below 3:1 contrast on the light surface, so the relief rule
applies and **every series carries a visible direct label**.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Headless-safe for script runs, but never clobbers a backend the caller
# already chose - a notebook that ran `%matplotlib inline` first keeps it,
# so the figures render inline instead of silently producing no output.
if matplotlib.get_backend().lower() in {"agg", ""}:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cwfc import frozen  # noqa: E402
from cwfc.attribution import (  # noqa: E402
    contribution_shares,
    cycle_jump_contributions,
    movement_summary,
    vm_timestamp_matrix,
    window_shift_contributions,
)
from cwfc.data import load_cpu_readings, load_vm_metadata  # noqa: E402
from cwfc.evaluate import (  # noqa: E402
    BASELINES,
    holdout_predictions,
    predictability_curve,
    rolling_origin_predictions,
)
from cwfc.models import CANDIDATES  # noqa: E402
from cwfc.policy import final_band_increments, label_buckets, select_nominal_q  # noqa: E402
from cwfc.seasonal import detect_period, phase_profile  # noqa: E402
from cwfc.targets import (  # noqa: E402
    AGGREGATE_CPU,
    build_targets,
    chronological_split,
    valid_reading_mask,
)

# --- palette (light mode) --------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # blue, orange, aqua

BUMP_START, BUMP_END = 53, 72
FIGSIZE = (9.0, 4.2)
DPI = 160


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Recessive chrome: hairline y-grid, no top/right spines, muted ticks."""
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=3)


def _figure():
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def _label_end(ax, x, y, text, color):
    """Direct label at a series' right end (the relief rule, and legibility)."""
    ax.annotate(
        text, xy=(x, y), xytext=(6, 0), textcoords="offset points",
        color=color, fontsize=9, fontweight="bold", va="center", ha="left",
    )


# --- context ---------------------------------------------------------------


@dataclass
class Context:
    """Everything the figures need, computed once."""

    y_train: np.ndarray
    period: int
    profile: object
    fold_preds: pd.DataFrame
    holdout: pd.DataFrame
    vm_mean: pd.Series
    band_increments: dict
    nominal_q: float
    movement: dict
    subscription_shares: dict


def load_context(data_dir: str = "challenge_data") -> Context:
    readings = load_cpu_readings(data_dir)
    meta = load_vm_metadata(data_dir)
    targets = build_targets(readings)
    train, _ = chronological_split(targets, holdout_size=frozen.HOLDOUT_SIZE)
    y = train[AGGREGATE_CPU].to_numpy(dtype=float)
    period = detect_period(y)

    fold_preds = rolling_origin_predictions(
        y, horizon=frozen.HOLDOUT_SIZE, step=frozen.ORIGIN_STEP,
        models={**BASELINES, **CANDIDATES},
    )
    selected_only = fold_preds[fold_preds["model"] == frozen.SELECTED_MODEL_NAME]
    q = select_nominal_q(selected_only, frozen.SELECTED_MODEL_NAME, frozen.TARGET_BREACH_RATE)

    train_ts = train["timestamp"].to_numpy()
    train_readings = readings[readings["timestamp"].isin(train_ts)]
    train_valid = train_readings.loc[valid_reading_mask(train_readings)]

    # Per-VM contributions to the two aggregate moves, for the net/gross figure.
    matrix = vm_timestamp_matrix(train_valid)
    effect = phase_profile(y, period).effect
    jump = cycle_jump_contributions(
        matrix, period, int(np.nanargmin(effect)), int(np.nanargmax(effect))
    )
    shift = window_shift_contributions(
        matrix, slice(BUMP_START, BUMP_END + 1), slice(0, BUMP_START)
    )
    contributions = {"hourly cycle": jump, "bump": shift}

    return Context(
        y_train=y,
        period=period,
        profile=phase_profile(y, period),
        movement={k: movement_summary(v) for k, v in contributions.items()},
        subscription_shares={
            k: contribution_shares(v, meta, "subscription_id") for k, v in contributions.items()
        },
        fold_preds=fold_preds,
        holdout=holdout_predictions(readings, frozen.HOLDOUT_SIZE),
        vm_mean=train_valid.groupby("vm_id")["avg_cpu"].mean(),
        band_increments=final_band_increments(
            selected_only, frozen.SELECTED_MODEL_NAME, q, monotone=frozen.BAND_MONOTONE
        ),
        nominal_q=q,
    )


# --- figures ---------------------------------------------------------------


def fig_training_series(ctx: Context):
    """Figure 1: the training series, with the bump and the slide marked."""
    y, P = ctx.y_train, ctx.period
    fig, ax = _figure()
    steps = np.arange(len(y))

    ax.axvspan(BUMP_START, BUMP_END, color=S2, alpha=0.10, zorder=1)
    dips = [i for i in range(len(y)) if i % P == int(np.nanargmin(ctx.profile.effect))]
    ax.scatter(dips, y[dips], s=26, facecolor=SURFACE, edgecolor=S2,
               linewidth=1.6, zorder=4, label="_nolegend_")
    ax.plot(steps, y, color=S1, linewidth=2.0, zorder=3)
    ax.axhline(y.mean(), color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)

    top = y.max() + 0.30
    ax.annotate(f"bump, steps {BUMP_START}-{BUMP_END} (mean {y[BUMP_START:BUMP_END+1].mean():.2f})",
                xy=(62, y[BUMP_START:BUMP_END + 1].max() + 0.02), xytext=(21, top - 0.09),
                color=INK_2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1))
    ax.annotate("post-bump slide", xy=(95, y[95]), xytext=(86, 9.76),
                color=INK_2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1))
    # Labels live in a right-hand margin and below the data floor, so the
    # series never runs through them.
    ax.annotate(f"training mean {y.mean():.2f}%", xy=(len(y), y.mean()),
                xytext=(6, 0), textcoords="offset points",
                color=MUTED, fontsize=8.5, ha="left", va="center")
    floor = y.min() - 0.20
    ax.annotate("hollow marks: the hourly dip phase", xy=(2, floor + 0.03),
                color=S2, fontsize=8.5)

    _style(ax, f"Fleet CPU utilisation moves inside a {y.max()-y.min():.2f}-point band",
           "training step (5 min apart)", "aggregate CPU utilisation (%)")
    ax.set_xlim(-2, len(y) + 20)
    ax.set_ylim(floor, top)
    fig.tight_layout()
    return fig


def fig_phase_profile(ctx: Context):
    """Figure 2: the hourly phase profile."""
    effect, P = ctx.profile.effect, ctx.period
    dip, rebound = int(np.nanargmin(effect)), int(np.nanargmax(effect))
    fig, ax = _figure()

    ax.bar(np.arange(P), effect, width=0.68, color=S1, zorder=3)
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=2)
    for phase, text in ((dip, f"dip {effect[dip]:+.3f}"), (rebound, f"rebound {effect[rebound]:+.3f}")):
        offset = -14 if effect[phase] < 0 else 8
        ax.annotate(text, xy=(phase, effect[phase]), xytext=(0, offset),
                    textcoords="offset points", ha="center", color=INK,
                    fontsize=9, fontweight="bold")

    _style(ax, f"The hourly cycle: a one-step dip and rebound, "
               f"{ctx.profile.amplitude:.2f} points peak-to-trough",
           f"phase (step mod {P}; one cycle = {P * 5} minutes)",
           "mean detrended deviation (points)")
    ax.set_xticks(np.arange(P))
    ax.set_ylim(min(effect) - 0.075, max(effect) + 0.055)
    fig.tight_layout()
    return fig


def fig_skill_by_horizon(ctx: Context):
    """Figure 3: skill against the training mean, by horizon step."""
    curve = predictability_curve(ctx.fold_preds).pivot(
        index="step", columns="model", values="skill_vs_train_mean"
    )
    fig, ax = _figure()
    series = [
        ("persistence", S1, "persistence"),
        ("last_cycle_mean", S2, "last-cycle mean"),
        (frozen.SELECTED_MODEL_NAME, S3, "profile + train mean"),
    ]
    ax.axhline(0, color=AXIS, linewidth=1.2, zorder=2)
    for name, color, label in series:
        ax.plot(curve.index, curve[name], color=color, linewidth=2.0, zorder=3, label=label)
        _label_end(ax, curve.index[-1], curve[name].iloc[-1], label, color)

    skill = curve["persistence"]
    crossing = int(skill[skill <= 0].index[0])
    ax.axvline(crossing, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(f"persistence loses its edge\nat step {crossing} (~{crossing * 5} min)",
                xy=(crossing, 0.30), xytext=(crossing + 1.2, 0.33),
                color=INK_2, fontsize=9)
    _style(ax, "Recent history stops paying after about 40 minutes",
           "horizon step (5 min apart)", "skill vs the training mean")
    lowest = float(np.nanmin(curve[[name for name, _, _ in series]].to_numpy()))
    ax.set_xlim(0.5, len(curve) + 4.6)
    ax.set_ylim(lowest - 0.34, float(np.nanmax(curve.to_numpy())) + 0.12)
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK_2, ncols=3)
    fig.tight_layout()
    return fig


def fig_holdout(ctx: Context):
    """Figure 4: the holdout - actual, frozen forecast with band, and the winner."""
    scored = ctx.holdout
    cpu = scored[scored["target"] == AGGREGATE_CPU]
    actual = cpu[cpu["model"] == "persistence"].sort_values("step")
    frozen_fc = cpu[cpu["model"] == frozen.SELECTED_MODEL_NAME].sort_values("step")
    winner = cpu[cpu["model"] == "last_cycle_mean"].sort_values("step")

    buckets = label_buckets(frozen_fc.assign(model=frozen.SELECTED_MODEL_NAME))
    increments = buckets["bucket"].map(ctx.band_increments).astype(float).to_numpy()
    steps = actual["step"].to_numpy()
    forecast = frozen_fc["y_pred"].to_numpy()

    truth = actual["y_true"].to_numpy()
    rival = winner["y_pred"].to_numpy()

    fig, ax = _figure()
    # Labels are attached to their own artists so the legend cannot drift out
    # of sync with what is drawn.
    ax.fill_between(steps, forecast, forecast + increments, color=S2, alpha=0.13,
                    zorder=1, linewidth=0, label=f"capacity band (q={ctx.nominal_q:.3f})")
    ax.plot(steps, forecast + increments, color=S2, linewidth=1.0,
            linestyle=(0, (3, 2)), zorder=2, label="_nolegend_")
    ax.plot(steps, rival, color=S3, linewidth=2.0, zorder=3,
            label="last-cycle mean, MAE 0.123")
    ax.plot(steps, forecast, color=S2, linewidth=2.0, zorder=4,
            label="frozen model, MAE 0.161")
    ax.plot(steps, truth, color=S1, linewidth=2.4, zorder=5, label="actual")

    ends = sorted(
        [(truth[-1], "actual", S1), (forecast[-1], "frozen model", S2),
         (rival[-1], "last-cycle mean", S3)],
        key=lambda item: item[0],
    )
    spacing = 0.055
    placed: list[float] = []
    for value, text, color in ends:
        position = value if not placed else max(value, placed[-1] + spacing)
        placed.append(position)
        _label_end(ax, steps[-1], position, text, color)

    bottom = min(truth.min(), rival.min(), forecast.min()) - 0.46
    ax.annotate("frozen model above actual\nat 20 of 24 steps (bias +0.145)",
                xy=(11, (forecast[10] + truth[10]) / 2), xytext=(1.0, bottom + 0.06),
                color=INK_2, fontsize=9, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1))

    _style(ax, "Holdout: the frozen model over-forecast a continuing decline",
           "holdout step (5 min apart)", "aggregate CPU utilisation (%)")
    ax.set_xlim(0.3, len(steps) + 6.4)
    ax.set_ylim(bottom, max((forecast + increments).max(), truth.max()) + 0.06)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5, labelcolor=INK_2, ncols=2)
    fig.tight_layout()
    return fig


def fig_concentration(ctx: Context):
    """Figure 5: how concentrated CPU demand is across VMs."""
    values = np.sort(ctx.vm_mean.to_numpy())[::-1]
    share_vms = np.arange(1, len(values) + 1) / len(values)
    share_cpu = np.cumsum(values) / values.sum()

    fig, ax = _figure()
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.plot(share_vms, share_cpu, color=S1, linewidth=2.2, zorder=3)
    ax.fill_between(share_vms, share_vms, share_cpu, color=S1, alpha=0.09, zorder=1, linewidth=0)

    for fraction in (0.10, 0.25):
        index = int(round(fraction * len(values))) - 1
        y_value = share_cpu[index]
        ax.scatter([fraction], [y_value], s=34, color=S1, zorder=4,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(f"top {fraction:.0%} of VMs carry {y_value:.0%}",
                    xy=(fraction, y_value), xytext=(10, -4), textcoords="offset points",
                    color=INK, fontsize=9, fontweight="bold")
    # Sits below the diagonal, in the empty half, so it never crosses the line.
    ax.annotate("if every VM used the same CPU", xy=(0.70, 0.46),
                color=MUTED, fontsize=8.5, ha="left")

    _style(ax, "A tenth of the fleet carries nearly half the load",
           "share of VMs, hottest first", "cumulative share of total CPU")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    return fig


def fig_net_vs_gross(ctx: Context):
    """Figure 6: the aggregate move is a residual of much larger activity."""
    events = list(ctx.movement)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.2, 4.3), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)

    # Left: gross movement against the net move it nets out to.
    positions = np.arange(len(events))
    width = 0.34
    n_vms = len(ctx.vm_mean)
    gross = [ctx.movement[e]["gross"] / n_vms for e in events]
    net = [abs(ctx.movement[e]["net"]) / n_vms for e in events]
    bar_gross = left.bar(positions - width / 2, gross, width, color=S2, zorder=3,
                         label="gross movement (sum of |per-VM change|)")
    bar_net = left.bar(positions + width / 2, net, width, color=S1, zorder=3,
                       label="net move (what the aggregate shows)")
    for index, (g, n) in enumerate(zip(gross, net)):
        left.annotate(f"{g:.2f}", xy=(index - width / 2, g), xytext=(0, 4),
                      textcoords="offset points", ha="center", color=S2,
                      fontsize=9, fontweight="bold")
        left.annotate(f"{n:.2f}", xy=(index + width / 2, n), xytext=(0, 4),
                      textcoords="offset points", ha="center", color=S1,
                      fontsize=9, fontweight="bold")
        left.annotate(f"{g / n:.1f}x", xy=(index, max(g, n)), xytext=(0, 24),
                      textcoords="offset points", ha="center", color=INK_2, fontsize=9)
    _style(left, "Hourly cycle and bump: each move is\na small residual",
           "", "CPU points, summed over VMs / n")
    left.set_xticks(positions)
    left.set_xticklabels(events)
    left.set_ylim(0, max(gross) * 1.34)

    # Right: subscriptions measured two ways - bump only. The hourly cycle's
    # top-25 shares (62% net / 38% gross) are in the report, not this panel.
    ks = [1, 5, 10, 25]
    shares = ctx.subscription_shares["bump"]
    net_share = [float(shares["share_of_net"].head(k).sum()) for k in ks]
    gross_share = [
        float(shares.sort_values("share_of_gross", ascending=False)["share_of_gross"].head(k).sum())
        for k in ks
    ]
    positions = np.arange(len(ks))
    right.bar(positions - width / 2, gross_share, width, color=S2, zorder=3)
    right.bar(positions + width / 2, net_share, width, color=S1, zorder=3)
    for index, (g, n) in enumerate(zip(gross_share, net_share)):
        right.annotate(f"{g:.0%}", xy=(index - width / 2, g), xytext=(0, 4),
                       textcoords="offset points", ha="center", color=S2,
                       fontsize=9, fontweight="bold")
        right.annotate(f"{n:.0%}", xy=(index + width / 2, n), xytext=(0, 4),
                       textcoords="offset points", ha="center", color=S1,
                       fontsize=9, fontweight="bold")
    right.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    right.annotate("100% of the net bump", xy=(-0.45, 1.0), xytext=(0, 6),
                   textcoords="offset points", ha="left", color=MUTED, fontsize=8.5)
    _style(right, "Bump only: concentration depends\non which you measure",
           "top-k subscriptions, bump", "share of the bump")
    right.set_xticks(positions)
    right.set_xticklabels([f"top {k}" for k in ks])
    right.set_ylim(0, max(max(net_share), 1.0) * 1.22)
    right.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    # One legend for both panels: the two colours mean the same thing in each.
    fig.legend(handles=[bar_gross, bar_net], frameon=False, fontsize=9,
               labelcolor=INK_2, loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


FIGURES = {
    "fig1_training_series": fig_training_series,
    "fig2_phase_profile": fig_phase_profile,
    "fig3_skill_by_horizon": fig_skill_by_horizon,
    "fig4_holdout": fig_holdout,
    "fig5_concentration": fig_concentration,
    "fig6_net_vs_gross": fig_net_vs_gross,
}


def save_all(outdir: str | Path = "reports/figures", ctx: Context | None = None) -> list[Path]:
    context = ctx or load_context()
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    written = []
    for name, builder in FIGURES.items():
        figure = builder(context)
        target = path / f"{name}.png"
        figure.savefig(target, facecolor=SURFACE, bbox_inches="tight")
        plt.close(figure)
        written.append(target)
        print(f"wrote {target}")
    return written


if __name__ == "__main__":
    save_all()
