"""Reproduce a stored anomaly development run exactly (m4_plan.md §1, "How v2 is built").

v1's code freeze (``2dc58fa``) is carried by behaviour once the frozen modules are
generalised behind a recipe version. This module recomputes a development run with the
current code, stores nothing, and compares everything the run stored:

* every column of ``intervals.parquet`` (scores, p-values, window sizes, tiers, flags);
* the injection results, the month-model ranges and the real-data cut checks (timings
  excepted);
* every event: its span, peak, tier, detectors, lowest p, categories, evidence and summary;
* the run record's counts, inputs, A4 counts, A5 results and E1 gate.

Values must be equal exactly (NaN equal to NaN). Nothing is written.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy import Engine

from gridcast.anomalies.dev import (
    compute_dev,
    cut_frame,
    forest_frame,
    interval_frame,
    run_config,
)
from gridcast.anomalies.freeze import freeze_state
from gridcast.anomalies.recipe import recipe_for
from gridcast.anomalies.report import StoredRun, load_run
from gridcast.config import AppConfig

# Run-record sections compared (timings, the freeze record and file lists are not results).
RECORD_KEYS = ("split", "region", "periods", "series", "inputs", "counts", "a5_real", "a4")
EVENT_COLUMNS = (
    "first_interval_end",
    "last_interval_end",
    "peak_interval_end",
    "tier",
    "detectors",
    "min_p",
    "categories",
    "lines_only",
    "evidence",
    "summary",
)


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    items: int  # rows or values compared
    differences: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.differences


@dataclass(slots=True)
class Reproduction:
    version: int
    run_id: int
    batch_id: str
    sha: str | None
    dirty: bool
    checks: list[Check]
    seconds: float

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


def frame_check(name: str, got: pd.DataFrame, stored: pd.DataFrame) -> Check:
    """Same columns (in order), same dtypes, same values; NaN equals NaN."""
    diffs: list[str] = []
    if list(got.columns) != list(stored.columns):
        diffs.append(f"columns {list(got.columns)} != stored {list(stored.columns)}")
        return Check(name, len(stored), tuple(diffs))
    if len(got) != len(stored):
        diffs.append(f"{len(got)} rows != stored {len(stored)}")
        return Check(name, len(stored), tuple(diffs))
    got, stored = got.reset_index(drop=True), stored.reset_index(drop=True)
    for c in stored.columns:
        a, b = got[c], stored[c]
        if a.dtype != b.dtype:
            diffs.append(f"{c}: dtype {a.dtype} != stored {b.dtype}")
        elif not a.equals(b):
            bad = ~((a == b) | (a.isna() & b.isna()))
            first = int(bad.to_numpy().argmax())
            diffs.append(f"{c}: {int(bad.sum())} rows differ (first at row {first})")
    return Check(name, len(stored), tuple(diffs))


def _json(value: Any) -> Any:
    """The value as the database returns it after a JSON round trip."""
    return json.loads(json.dumps(value, default=str, allow_nan=False))


def event_frame(out: Any) -> pd.DataFrame:
    """The recomputed events in the stored table's columns and order."""
    ends = out.series.ends
    rows = [
        {
            "first_interval_end": ends[e.first],
            "last_interval_end": ends[e.last],
            "peak_interval_end": ends[e.peak],
            "tier": e.tier,
            "detectors": list(e.detectors),
            "min_p": e.min_p,
            "categories": list(e.categories),
            "lines_only": e.lines_only,
            "evidence": _json(ev),
            "summary": summary,
        }
        for e, ev, summary in zip(
            out.detection.events, out.evidence.evidence, out.evidence.summaries, strict=True
        )
    ]
    return pd.DataFrame(rows, columns=list(EVENT_COLUMNS))


def events_check(got: pd.DataFrame, stored: pd.DataFrame) -> Check:
    diffs: list[str] = []
    stored = stored[list(EVENT_COLUMNS)].reset_index(drop=True)
    if len(got) != len(stored):
        return Check("events", len(stored), (f"{len(got)} events != stored {len(stored)}",))
    for c in EVENT_COLUMNS:
        bad = [i for i, (a, b) in enumerate(zip(got[c], stored[c], strict=True)) if not _same(a, b)]
        if bad:
            diffs.append(f"{c}: {len(bad)} events differ (first at event {bad[0]})")
    return Check("events", len(stored), tuple(diffs))


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b)
    if isinstance(a, pd.Timestamp) or isinstance(b, pd.Timestamp):
        return bool(pd.Timestamp(a) == pd.Timestamp(b))
    return bool(a == b)


def record_check(got: dict[str, Any], stored: dict[str, Any]) -> Check:
    got = _json(got)
    diffs = tuple(
        f"{k}: {got.get(k)!r} != stored {stored.get(k)!r}"
        for k in RECORD_KEYS
        if got.get(k) != stored.get(k)
    )
    gate_got, gate_stored = got.get("shap_gate", {}), stored.get("shap_gate", {})
    if gate_got != gate_stored:
        diffs += (f"shap_gate: {gate_got!r} != stored {gate_stored!r}",)
    return Check("run record", len(RECORD_KEYS) + 1, diffs)


def without_seconds(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["seconds"])


def reproduce(engine: Engine, cfg: AppConfig, region_id: str, version: int = 1) -> Reproduction:
    """Recompute recipe ``version``'s stored development run and compare everything
    (nothing is written)."""
    t = time.perf_counter()
    stored: StoredRun = load_run(engine, cfg, region_id, version)
    state = freeze_state(cfg, version=version)
    out = compute_dev(engine, cfg, region_id, recipe_for(cfg, version))
    intervals = interval_frame(out.detection, out.split)
    checks = [
        frame_check("intervals.parquet", intervals, stored.intervals),
        frame_check("injections.parquet", out.injections, stored.injections),
        frame_check("forests.parquet", forest_frame(out.detection), stored.forests),
        frame_check(
            "cuts.parquet", without_seconds(cut_frame(out.cuts)), without_seconds(stored.cuts)
        ),
        events_check(event_frame(out), stored.events),
        record_check(run_config(out, cfg, state, region_id), stored.config),
    ]
    return Reproduction(
        version,
        stored.run_id,
        stored.batch_id,
        state.sha,
        state.dirty,
        checks,
        time.perf_counter() - t,
    )


def summary_lines(rep: Reproduction) -> list[str]:
    head = (
        f"recipe v{rep.version}: run {rep.run_id} (batch {rep.batch_id}) recomputed at {rep.sha}"
        f"{' (dirty tree)' if rep.dirty else ''} in {rep.seconds:.0f} s"
    )
    lines = [head]
    for c in rep.checks:
        status = "identical" if c.ok else "DIFFERS"
        lines.append(f"   {c.name}: {c.items:,} compared, {status}")
        lines += [f"      {d}" for d in c.differences]
    lines.append("reproduced exactly" if rep.ok else "NOT reproduced")
    return lines
