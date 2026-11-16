#!/usr/bin/env python3
"""Build and execute notebooks/workload_analysis.ipynb from current code.

The notebook is a thin viewing layer over analysis/figures.py. This script is
its single source: it writes the cells below, executes them against the
current code and data, and saves the notebook with fresh outputs. Never edit
the .ipynb by hand; edit the cells here and rerun.

    python scripts/build_notebook.py [--output PATH] [--timeout SECONDS]

Run from the repository root with challenge_data/ extracted. The kernel runs
inside notebooks/, which is what the relative paths in the cells assume.
Cell ids are fixed so a rebuild diffs only where content or outputs changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "notebooks" / "workload_analysis.ipynb"

# (cell type, fixed id, source)
CELLS = [
    ('markdown', '0e0cb5fe', """\
# Workload analysis

A thin viewing layer. Every number and figure here is produced by the project's
own modules — this notebook imports them rather than re-deriving anything, so it
cannot drift from `reports/`.

- Analysis prose and tables: [`reports/`](../reports/)
- Decisions, with alternatives and outcomes: [`DECISIONS.md`](../DECISIONS.md)
- Final report: [`reports/final_report.md`](../reports/final_report.md)

Run from inside `notebooks/` (where Jupyter opens it), with `challenge_data/`
extracted at the repository root: the paths below are relative to `notebooks/`."""),
    ('code', 'a097c878', """\
%matplotlib inline
import sys
sys.path.insert(0, "../analysis")

import figures

ctx = figures.load_context("../challenge_data")
print(f"training steps : {len(ctx.y_train)}")
print(f"detected period: {ctx.period} steps = {ctx.period * 5} minutes")
increments = {k: round(v, 4) for k, v in ctx.band_increments.items()}
print(f"band quantile  : {ctx.nominal_q:.3f}")
print(f"band increments: {increments}")"""),
    ('markdown', '5b0b2231', """\
## 1. The training series

The whole fleet moves inside a 0.8-point band. The shaded window is the bump
(steps 53–72); the hollow marks are the recurring hourly dip."""),
    ('code', 'f9ad53e2', """\
figures.fig_training_series(ctx);"""),
    ('markdown', 'e19283d3', """\
## 2. The hourly cycle

A one-step dip and rebound, 0.34 points peak-to-trough — about 2.8× a typical
step change. Subtracting this profile halves step-to-step volatility and lifts
lag-1 autocorrelation from 0.662 to 0.879."""),
    ('code', 'ab9aa971', """\
figures.fig_phase_profile(ctx);"""),
    ('markdown', '6ecf242d', """\
## 3. How far ahead is this workload predictable?

Skill against the long-run training mean, by horizon step, across the 13
expanding-window folds. Persistence loses its edge at step 9."""),
    ('code', 'bd7606ed', """\
figures.fig_skill_by_horizon(ctx);"""),
    ('markdown', '5a85ac28', """\
## 4. The holdout

The frozen model over-forecast a continuing decline and placed fourth of five.
`DECISIONS.md` entry 22, written before the holdout was scored, predicted this
failure. The band never breached — because the point forecast was biased high,
which makes 0/24 uninformative about the band's calibration."""),
    ('code', 'b524fa55', """\
figures.fig_holdout(ctx);"""),
    ('markdown', 'cc19259e', """\
## 5. Demand concentration

Capacity risk lives in the hot tail, not the fleet mean: ~2% of VMs run above
50% utilisation while the fleet sits at 9.4%."""),
    ('code', '9fb70e08', """\
figures.fig_concentration(ctx);"""),
    ('markdown', '4be3abfb', """\
## 6. Net versus gross movement

Each aggregate move is a small residual of movement about four times larger
underneath it, and roughly half the fleet moves *against* the aggregate each
time. Measuring concentration on the net move alone overstates it three- to
fourfold."""),
    ('code', 'c6c6148f', """\
figures.fig_net_vs_gross(ctx);"""),
    ('markdown', '81215005', """\
---

### Reproducing everything

```bash
python -m pytest -q
python -m cwfc.predict --data challenge_data --output predictions.csv
python -m cwfc.score_holdout --data challenge_data --predictions predictions.csv
python analysis/01_data_and_workload.py
python analysis/02_cycle_attribution.py
python analysis/03_validation_and_policy.py
python analysis/04_core_weighted.py
python analysis/figures.py
```"""),
]


def build() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    makers = {"markdown": nbformat.v4.new_markdown_cell, "code": nbformat.v4.new_code_cell}
    nb.cells = [makers[kind](source.rstrip("\n"), id=cell_id) for kind, cell_id, source in CELLS]
    return nb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=600, help="per-cell timeout, seconds")
    args = parser.parse_args()

    nb = build()
    # Execute with notebooks/ as the working directory, matching how Jupyter
    # opens the committed file. Errors raise, so a broken cell fails the build.
    NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(REPO_ROOT / "notebooks")}},
    ).execute()
    nbformat.validate(nb)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.output)
    print(f"wrote {args.output} ({len(nb.cells)} cells, executed)")


if __name__ == "__main__":
    main()
