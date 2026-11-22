"""The holdout anomaly run, scored once per recipe version (m4_plan.md step 6, A6;
``gridcast evaluate-holdout-anomalies --version 1|2``).

The readings were fixed before coding (m4_plan.md §1, "Clarified in step 6"):

* **Refusals first** (:class:`HoldoutRefused`; nothing is computed): the version's freeze
  must hold on a clean tree; its development run (run 104 for v1, run 106 for v2) must be
  stored; **the A6 lock:** no successful holdout run may be stored for the recipe hash; and
  the two QR batches must agree on their overlap (:func:`~gridcast.anomalies.inputs.split_band`).
* **Reproduction before scoring** (:class:`HoldoutReproductionFailed`; nothing is stored):
  1. :func:`~gridcast.anomalies.reproduce.reproduce`, the full check of the version's
     development run (for v1 also the check that carries its code freeze);
  2. one continuous run over the whole series (data up to NEM midnight of
     ``backtest.holdout_end``). Its development part must equal the stored development
     batch exactly: per-interval scores, p-values, window sizes, tiers and flags under the
     development split, the month models fitted before the holdout, and the development
     events its alerts give (:func:`development_part`).
  Until both pass, nothing about the holdout part is printed, logged or stored.
* **Then the holdout split** (NEM days ``holdout_start`` → ``holdout_end``, every detector):
  its tiers and events, the injections (§2.9, starts drawn from the holdout's scored
  intervals), A5's real cuts inside the holdout and the month-model leak check, and the
  evidence of every holdout event with SHAP at demand peaks behind E1 (a gate failure stops
  the run before anything is stored).
* **Storage** as the development run (:func:`~gridcast.anomalies.dev.store_dev`), after the
  lock is checked again: one ``model_runs`` row with split ``holdout``, parquet under
  ``<data root>/anomalies/<batch>/`` (``intervals.parquet`` holds the whole continuous run
  with the holdout's scored-period flags), and the events in ``anomalies``. The run record
  adds the reproduction (``reproduction``).

The criteria are computed by ``gridcast report-m4-holdout`` from the stored runs.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from gridcast.anomalies.dev import (
    CutCheck,
    DevOutputs,
    DevRun,
    existing_runs,
    forest_frame,
    interval_frame,
    run_config,
    store_dev,
)
from gridcast.anomalies.dev import summary_lines as dev_summary_lines
from gridcast.anomalies.evidence_inputs import run_evidence
from gridcast.anomalies.freeze import FreezeState, freeze_state
from gridcast.anomalies.injection import run_injections
from gridcast.anomalies.inputs import check_band_actuals, load_market_series, split_band
from gridcast.anomalies.recipe import (
    cut_indices,
    dev_split,
    differences,
    forest_leaks,
    holdout_split,
    recipe_for,
    truncated_detection,
)
from gridcast.anomalies.report import StoredRun, load_run
from gridcast.anomalies.reproduce import Check, Reproduction, frame_check, reproduce, same_value
from gridcast.anomalies.run import Detection, Split, alert_tiers, detect, split_events
from gridcast.anomalies.series import MarketSeries, QuantileBand
from gridcast.config import AppConfig
from gridcast.explainability.stored import stored_fits
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.logs import get_logger

log = get_logger(__name__)

# Event columns compared on the development part (evidence and summaries are compared by
# the full reproduction, which rebuilds them from the development data alone).
DEV_EVENT_COLUMNS = (
    "first_interval_end",
    "last_interval_end",
    "peak_interval_end",
    "tier",
    "detectors",
    "min_p",
    "categories",
    "lines_only",
)


class HoldoutRefused(RuntimeError):
    """A precondition of the holdout run is not met; nothing was run."""


class HoldoutReproductionFailed(RuntimeError):
    """The development part did not reproduce; nothing was stored."""


# --------------------------------------------------------------------------- refusals


@dataclass(frozen=True, slots=True)
class Preflight:
    version: int
    state: FreezeState
    dev_run_id: int
    dev_batch_id: str


def preflight(engine: Engine, cfg: AppConfig, region_id: str, version: int) -> Preflight:
    """The refusals that need no data: freeze, clean tree, development run stored, A6 lock."""
    state = freeze_state(cfg, version=version)
    problems = list(state.problems)
    if state.sha is None:
        problems.append("not a git checkout")
    if state.dirty:
        problems.append("the working tree has uncommitted changes to tracked files")
    dev = existing_runs(engine, region_id, "dev", state.config_hash)
    if len(dev) != 1:
        problems.append(
            f"expected one development run for hash {state.config_hash}, found {len(dev)}"
        )
    held = existing_runs(engine, region_id, "holdout", state.config_hash)
    if held:
        problems.append(
            f"a holdout run is already stored for hash {state.config_hash} (run {held[0][0]}, "
            f"batch {held[0][1]}): the holdout is scored once (A6)"
        )
    if problems:
        raise HoldoutRefused("; ".join(problems))
    return Preflight(version, state, int(dev[0][0]), str(dev[0][1]))


# --------------------------------------------------------------------------- inputs


@dataclass(slots=True)
class HoldoutInputs:
    series: MarketSeries
    band: QuantileBand
    checked_actuals: int
    holdout_fits: int


def load_inputs(engine: Engine, cfg: AppConfig, region_id: str) -> HoldoutInputs:
    """The continuous series to the holdout end and the joined band (after the QR overlap
    check), with the stored actuals checked against market demand. Nothing is scored."""
    last = nem_midnight_utc(str(cfg.backtest.holdout_end))
    series = load_market_series(engine, cfg, region_id, last_end=last)
    if series.ends[-1] != last:
        raise HoldoutRefused(f"the market series ends at {series.ends[-1]}, not at {last}")
    try:
        band, actual = split_band(cfg, "holdout")
    except ValueError as exc:
        raise HoldoutRefused(f"the holdout band: {exc}") from exc
    band = band.issued_until(last)
    actual = actual[: len(band.issue_times)]
    checked = check_band_actuals(band, actual, series)
    fits = stored_fits(engine, cfg, "holdout")
    return HoldoutInputs(series, band, checked, len(fits))


# --------------------------------------------------------------------------- reproduction


def development_part(
    full: Detection,
    series: MarketSeries,
    dev: Split,
    stored: StoredRun,
    cfg: AppConfig,
) -> list[Check]:
    """The continuous run's development part against the stored development batch.

    Its alerts are tiered and merged under the development split, as the development run
    did; the per-interval frame, the month models before the holdout and the events must
    equal the stored ones exactly.
    """
    a = cfg.anomalies
    tiers = alert_tiers(full.run, series.price, dev, a)
    events = split_events(full.run, tiers, series.price, dev, a)
    view = Detection(full.run, tiers, events)
    n = len(stored.intervals)
    checks: list[Check] = []

    if n > len(series) or not series.ends[:n].equals(
        pd.DatetimeIndex(stored.intervals["interval_end"])
    ):
        return [Check("development grid", n, ("the interval grid differs from the stored",))]
    got = interval_frame(view, dev).iloc[:n].reset_index(drop=True)
    checks.append(frame_check("development intervals", got, stored.intervals))

    months = forest_frame(view)
    boundary = nem_midnight_utc(str(dev.end_day))
    before = months[pd.to_datetime(months["month_start"], utc=True) < boundary]
    checks.append(
        frame_check("development month models", before.reset_index(drop=True), stored.forests)
    )

    ends = series.ends
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
        }
        for e in events
    ]
    checks.append(event_check(pd.DataFrame(rows, columns=list(DEV_EVENT_COLUMNS)), stored.events))
    return checks


def event_check(got: pd.DataFrame, stored: pd.DataFrame) -> Check:
    stored = stored[list(DEV_EVENT_COLUMNS)].reset_index(drop=True)
    if len(got) != len(stored):
        return Check("development events", len(stored), (f"{len(got)} events != {len(stored)}",))
    diffs = []
    for c in DEV_EVENT_COLUMNS:
        bad = [
            i
            for i, (x, y) in enumerate(zip(got[c], stored[c], strict=True))
            if not same_value(x, y)
        ]
        if bad:
            diffs.append(f"{c}: {len(bad)} events differ (first at event {bad[0]})")
    return Check("development events", len(stored), tuple(diffs))


# --------------------------------------------------------------------------- the run


@dataclass(slots=True)
class HoldoutRun:
    preflight: Preflight
    reproduction: Reproduction
    development: list[Check]
    outputs: DevOutputs
    seconds: dict[str, float] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        """The run-record section that documents the reproduction."""
        rep = self.reproduction
        return {
            "reproduction": {
                "development_run": rep.run_id,
                "development_batch": rep.batch_id,
                "full": [{"name": c.name, "items": c.items} for c in rep.checks],
                "development_part": [{"name": c.name, "items": c.items} for c in self.development],
                "seconds": round(rep.seconds, 1),
            }
        }


def _failed(checks: list[Check]) -> list[str]:
    return [f"{c.name}: {'; '.join(c.differences)}" for c in checks if not c.ok]


def compute_holdout(engine: Engine, cfg: AppConfig, region_id: str, pre: Preflight) -> HoldoutRun:
    """Reproduce, then score the holdout (no writes)."""
    a = cfg.anomalies
    recipe = recipe_for(cfg, pre.version)
    secs: dict[str, float] = {}

    t = time.perf_counter()
    rep = reproduce(engine, cfg, region_id, pre.version)
    secs["reproduce"] = time.perf_counter() - t
    if rep.run_id != pre.dev_run_id:
        raise HoldoutReproductionFailed(
            f"reproduced run {rep.run_id}, but the stored development run is {pre.dev_run_id}"
        )
    if not rep.ok:
        raise HoldoutReproductionFailed(
            "the development run did not reproduce: " + " | ".join(_failed(rep.checks))
        )
    log.info("anomalies.holdout.reproduced", run=rep.run_id, seconds=round(secs["reproduce"], 1))

    t = time.perf_counter()
    inputs = load_inputs(engine, cfg, region_id)
    secs["load"] = time.perf_counter() - t

    t = time.perf_counter()
    split = holdout_split(cfg)
    det = detect(inputs.series, inputs.band, split, a, recipe=recipe)
    secs["detect"] = time.perf_counter() - t
    stored = load_run(engine, cfg, region_id, pre.version)
    dev_checks = development_part(det, inputs.series, dev_split(cfg), stored, cfg)
    if not all(c.ok for c in dev_checks):
        raise HoldoutReproductionFailed(
            "the continuous run's development part differs from the stored batch: "
            + " | ".join(_failed(dev_checks))
        )
    log.info("anomalies.holdout.development_part_equal", checks=len(dev_checks))

    t = time.perf_counter()
    inj = run_injections(inputs.series, det.run, split, a)
    secs["inject"] = time.perf_counter() - t

    cuts = []
    for c in cut_indices(inputs.series.ends, split, a):
        t = time.perf_counter()
        cut_run = truncated_detection(inputs.series, inputs.band, split, a, int(c), recipe=recipe)
        diff = differences(det, cut_run, int(c))
        cuts.append(CutCheck(int(c), inputs.series.ends[int(c)], diff, time.perf_counter() - t))
    secs["cuts"] = sum(cc.seconds for cc in cuts)
    leaks = forest_leaks(det)

    t = time.perf_counter()
    ev = run_evidence(engine, cfg, region_id, "holdout", inputs.series, det)
    secs["evidence"] = time.perf_counter() - t

    out = DevOutputs(
        split=split,
        series=inputs.series,
        detection=det,
        injections=inj,
        cuts=cuts,
        forest_leaks=leaks,
        evidence=ev,
        band_issue_times=len(inputs.band.issue_times),
        band_valid_pairs=int(inputs.band.valid.sum()),
        checked_actuals=inputs.checked_actuals,
        seconds={k: v for k, v in secs.items() if k != "reproduce"},
    )
    return HoldoutRun(pre, rep, dev_checks, out, secs)


def store_holdout(
    engine: Engine, cfg: AppConfig, region_id: str, run: HoldoutRun
) -> tuple[int, uuid.UUID, Path]:
    """Check the A6 lock again, then store as the development run does."""
    state = run.preflight.state
    held = existing_runs(engine, region_id, "holdout", state.config_hash)
    if held:
        raise HoldoutRefused(
            f"a holdout run was stored for hash {state.config_hash} meanwhile (run {held[0][0]})"
        )
    return store_dev(engine, cfg, region_id, run.outputs, state, extra_config=run.record())


@dataclass(slots=True)
class StoredHoldout:
    run_id: int
    batch_id: uuid.UUID
    directory: Path
    run: HoldoutRun
    config: dict


def run_holdout(engine: Engine, cfg: AppConfig, region_id: str, version: int) -> StoredHoldout:
    pre = preflight(engine, cfg, region_id, version)
    log.info("anomalies.holdout.start", version=version, hash=pre.state.config_hash)
    run = compute_holdout(engine, cfg, region_id, pre)
    t = time.perf_counter()
    run_id, batch_id, directory = store_holdout(engine, cfg, region_id, run)
    run.seconds["store"] = time.perf_counter() - t
    config = run_config(run.outputs, cfg, pre.state, region_id) | run.record()
    return StoredHoldout(run_id, batch_id, directory, run, config)


def summary_lines(stored: StoredHoldout) -> list[str]:
    """Counts, checks and timings only; the criteria are computed by ``report-m4-holdout``."""
    rep = stored.run.reproduction
    lines = [
        f"development run {rep.run_id} (batch {rep.batch_id}) reproduced exactly in "
        f"{rep.seconds:.0f} s: " + ", ".join(f"{c.name} ({c.items:,})" for c in rep.checks),
        "continuous run's development part equal to the stored batch: "
        + ", ".join(f"{c.name} ({c.items:,})" for c in stored.run.development),
    ]
    dev_run = DevRun(
        stored.run_id, stored.batch_id, stored.directory, stored.run.outputs, stored.config
    )
    lines += dev_summary_lines(dev_run)
    lines.append(
        "   total seconds: " + ", ".join(f"{k} {v:.0f}" for k, v in stored.run.seconds.items())
    )
    return lines


def check_lines(engine: Engine, cfg: AppConfig, region_id: str, pre: Preflight) -> list[str]:
    """``--check-only``: the refusals passed; load and check the inputs (nothing scored)."""
    inputs = load_inputs(engine, cfg, region_id)
    s, b = inputs.series, inputs.band
    return [
        f"recipe v{pre.version}: hash {pre.state.config_hash}, commit {pre.state.sha}, frozen "
        f"commit {pre.state.frozen_commit}; development run {pre.dev_run_id} (batch "
        f"{pre.dev_batch_id}); no holdout run stored",
        f"   series {len(s):,} intervals ({s.ends[0]} -> {s.ends[-1]}); band "
        f"{len(b.issue_times):,} issue times ({b.issue_times[0]} -> {b.issue_times[-1]}), "
        f"{int(np.sum(b.valid)):,} valid pairs; actuals equal market demand on "
        f"{inputs.checked_actuals:,} pairs; holdout_main fits {inputs.holdout_fits}",
    ]
