"""Shared helpers for the analysis scripts.

These scripts replace a notebook deliberately (ASSIGNMENT.md allows
"notebooks/ or equivalent analysis"): a script has no hidden execution state,
cannot ship stale outputs alongside changed code, and regenerates its own
markdown on every run. Each writes one file into reports/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPORTS = Path(__file__).resolve().parent.parent / "reports"


class Report:
    """Accumulates markdown and prints it as it goes."""

    def __init__(self, title: str, subtitle: str = "") -> None:
        self.lines: list[str] = []
        self.h1(title)
        if subtitle:
            self.text(subtitle)

    def _add(self, text: str) -> None:
        print(text)
        self.lines.append(text)

    def h1(self, text: str) -> None:
        self._add(f"\n# {text}\n")

    def h2(self, text: str) -> None:
        self._add(f"\n## {text}\n")

    def h3(self, text: str) -> None:
        self._add(f"\n### {text}\n")

    def text(self, text: str = "") -> None:
        self._add(text)

    def bullets(self, items) -> None:
        for item in items:
            self._add(f"- {item}")
        self._add("")

    def kv(self, pairs) -> None:
        for key, value in pairs:
            self._add(f"- **{key}**: {value}")
        self._add("")

    def table(self, frame: pd.DataFrame, index: bool = True, floatfmt: int = 4) -> None:
        rounded = frame.round(floatfmt) if hasattr(frame, "round") else frame
        self._add("```")
        self._add(rounded.to_string(index=index))
        self._add("```")
        self._add("")

    def note(self, text: str) -> None:
        """An operational note: what an infrastructure team does about it."""
        self._add(f"> **Operational note.** {text}")
        self._add("")

    def caveat(self, text: str) -> None:
        self._add(f"> **Caveat.** {text}")
        self._add("")

    def save(self, filename: str) -> Path:
        REPORTS.mkdir(parents=True, exist_ok=True)
        path = REPORTS / filename
        path.write_text("\n".join(self.lines).lstrip("\n") + "\n")
        print(f"\nwrote {path}")
        return path
