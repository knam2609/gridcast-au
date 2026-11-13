"""The holdout intervals, scored once (M3 step 6b, U5; ``gridcast evaluate-holdout-intervals``).

The second of step 6's two stages (m3_plan.md step 6). Everything that scores a holdout
interval happens here, under one lock.

* **Refusals first:** the interval freeze must hold (both recipe hashes, frozen code
  unchanged since ``intervals.frozen_commit``), the tree must be clean, and every input must
  be stored: run 2, M2's ``holdout_main`` (frozen M2 recipe), step 2's QR batch, step 6a's
  QR holdout batch (reproduced, built on those batches at the frozen commit), and the
  development interval batches of steps 1, 3 and 5a. **The U5 lock:** no holdout interval
  run may exist for the frozen interval recipe.
* **One continuous run.** The frozen recipe runs over issue times ``ml.first_issue`` →
  ``backtest.holdout_end``, as if deployed continuously: calibration windows, ACI state and
  selections carry across ``backtest.holdout_start``. Its point inputs are run 2's stored
  forecasts before the holdout joined to ``holdout_main``'s; its QR inputs are step 2's
  quantiles joined to step 6a's. M2's holdout run and step 6a both reproduced the last
  development retrain exactly, so the joined inputs equal one continuous walk-forward.
  The workers mirror :func:`~gridcast.intervals.batch.conformal_for`,
  :func:`~gridcast.intervals.cqr_batch.cqr_for` and the half-hour runs
  (:func:`~gridcast.intervals.halfhour.period_run`) on the joined inputs, with the frozen
  functions unchanged.
* **Reproduction before scoring:** the development part of every result (issue times from
  ``intervals.first_issue`` to the holdout) must equal the stored development batch exactly,
  bounds and selections: step 1's conformal runs, step 3's CQR and step 5a's half-hour runs.
  Step 6a's quantiles for the overlap (the last development retrain's fold) must equal step
  2's. Otherwise :class:`HoldoutIntervalsReproductionFailed` is raised; nothing is scored.
* **Scoring:** only then are the holdout issue times scored, on ``holdout_main``'s stored
  actuals and common mask, with the stored metrics of each development batch: 5-minute
  conformal around C and the baselines, CQR, raw QR (step 6a's quantiles) and the half-hour
  runs. Nothing is printed or logged; ``gridcast report-m3-holdout`` reads the stored runs.

Storage (:func:`store_holdout_intervals`, which checks the lock again first): one batch,
bounds of the holdout issue times under ``<data root>/<intervals.output_dir>/<batch id>/5min/``
and ``…/30min/`` (laid out as the development batches), one ``model_runs`` row per run
(evaluation set ``intervals_holdout``) and its metrics with split ``holdout``. Raw QR's row
points at step 6a's batch, which holds its quantiles.
"""

from __future__ import annotations

import gc
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Connection, Engine, text

from gridcast.config import AppConfig
from gridcast.evaluation.holdout import HoldoutRefused, HoldoutWindow, holdout_window, same_instants
from gridcast.forecasting.ml_backtest import ml_config_hash
from gridcast.forecasting.origins import ForecastSetup, nem_midnight_utc
from gridcast.intervals.aemo_report import find_halfhour_runs
from gridcast.intervals.batch import (
    TARGET,
    bounds_metrics,
    candidate_grid,
    check_source_batch,
    interval_groups,
    per_lead_rows,
    result_config,
    save_results,
    selection_retrains,
)
from gridcast.intervals.batch import run_name as conformal_run_name
from gridcast.intervals.conformal import (
    ConformalResult,
    point_problem,
    quantile_problem,
    run_conformal,
)
from gridcast.intervals.cqr_batch import RUN_NAME as CQR_RUN_NAME
from gridcast.intervals.cqr_batch import (
    QrSource,
    check_qr_batch,
    cqr_metrics,
    merge_levels,
    read_qr_quantiles,
)
from gridcast.intervals.freeze import FreezeState
from gridcast.intervals.halfhour import (
    CONFORMAL,
    CQR,
    CQR_NAME,
    CQR_POINT,
    PERIOD_MINUTES,
    PeriodTask,
    build_inputs,
    conformal_name,
    period_bands,
    period_run,
)
from gridcast.intervals.inputs import StoredBatch, read_stored_batch, read_wide
from gridcast.intervals.qr_batch import POINT_MODEL
from gridcast.intervals.qr_batch import RUN_NAME as QR_RUN_NAME
from gridcast.intervals.qr_holdout import CROSSED, PointBatch, holdout_point_batch
from gridcast.intervals.qr_holdout import EVALUATION_SET as QR_HOLDOUT_SET
from gridcast.intervals.quantile import interval_quantiles, quantile_key, quantile_levels
from gridcast.intervals.report import IntervalRun, find_runs
from gridcast.intervals.scores import pinball
from gridcast.logs import get_logger

log = get_logger(__name__)

EVALUATION_SET = "intervals_holdout"
SPLIT = "holdout"
RES_5, RES_30 = "5min", "30min"
SELECTION_FIELDS = ("retrain", "level", "band", "chosen", "n_pairs")


class HoldoutIntervalsRefused(RuntimeError):
    """A precondition of the single holdout scoring run is not met; nothing was run."""


class HoldoutIntervalsReproductionFailed(RuntimeError):
    """The continuous run does not reproduce the stored development outputs; nothing scored."""


# --------------------------------------------------------------------------- preconditions


@dataclass(frozen=True, slots=True)
class QrHoldoutBatch:
    """Step 6a's stored QR holdout batch."""

    batch_id: str
    run_id: int
    directory: Path
    config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class HoldoutInputs:
    window: HoldoutWindow
    freeze: FreezeState
    holdout_point: PointBatch
    dev_qr: QrSource
    qr_holdout: QrHoldoutBatch
    dev_runs: dict[str, IntervalRun]  # 5-minute development runs by name
    dev_halfhour: dict[str, IntervalRun]  # half-hour development runs by name


def stored_holdout_batches(engine: Engine, region_id: str, interval_hash: str) -> list[str]:
    """Holdout interval batches already stored for the interval recipe (the U5 lock)."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT batch_id::text FROM model_runs
                WHERE region_id = :r AND target = :t AND model_kind = 'interval'
                  AND config->>'evaluation_set' = :es
                  AND config->>'interval_config_hash' = :h
                """
            ),
            {"r": region_id, "t": TARGET, "es": EVALUATION_SET, "h": interval_hash},
        )
        return sorted(str(b) for b in rows.scalars())


def qr_holdout_batch(
    engine: Engine, cfg: AppConfig, region_id: str, point_batch: str
) -> QrHoldoutBatch:
    """Step 6a's batch for the frozen QR recipe, checked against the other inputs."""
    qr_hash = cfg.intervals.frozen_qr_config_hash
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT run_id, batch_id::text AS batch, config, artifact_path FROM model_runs
                WHERE region_id = :r AND model_name = :m AND config->>'evaluation_set' = :es
                  AND config->>'qr_config_hash' = :h AND status = 'success'
                """
            ),
            {"r": region_id, "m": QR_RUN_NAME, "es": QR_HOLDOUT_SET, "h": qr_hash},
        ).all()
    if len(rows) != 1:
        raise LookupError(
            f"expected one QR holdout batch for QR recipe {qr_hash}, found {len(rows)}: "
            "run gridcast backtest-qr-holdout first"
        )
    row = rows[0]
    config: dict[str, Any] = dict(row.config)
    problems = []
    if not config.get("complete_window"):
        problems.append("is not complete")
    if not config.get("qr_reproduction", {}).get("passed"):
        problems.append("did not reproduce step 2's QR batch")
    if not config.get("point_reproduction", {}).get("passed"):
        problems.append("did not reproduce the point forecasts")
    if config.get("development_qr_batch") != cfg.intervals.qr_batch:
        problems.append(f"continues QR batch {config.get('development_qr_batch')}")
    if config.get("source_batch") != cfg.intervals.source_batch:
        problems.append(f"was built on source batch {config.get('source_batch')}")
    if config.get("holdout_point_batch") != point_batch:
        problems.append(f"was built on holdout batch {config.get('holdout_point_batch')}")
    if config.get("frozen_commit") != cfg.intervals.frozen_commit:
        problems.append(f"was built under freeze {config.get('frozen_commit')}")
    if config.get("scored") is not False:
        problems.append("is not an unscored step-6a batch")
    if problems:
        raise ValueError(f"QR holdout batch {row.batch}: " + "; ".join(problems))
    return QrHoldoutBatch(
        str(row.batch), int(row.run_id), cfg.data_root() / row.artifact_path, config
    )


def preflight(engine: Engine, cfg: AppConfig, region_id: str, freeze: FreezeState) -> HoldoutInputs:
    """Every refusal check, the U5 lock included; returns what the run reads."""
    problems = list(freeze.problems)
    if freeze.sha is None:
        problems.append("not a git checkout (the run's commit cannot be recorded)")
    elif freeze.dirty:
        problems.append("the working tree has uncommitted changes")
    window: HoldoutWindow | None = None
    try:
        window = holdout_window(cfg)
    except HoldoutRefused as exc:
        problems.append(str(exc))
    try:
        check_source_batch(engine, cfg.intervals.source_batch)
    except (LookupError, ValueError) as exc:
        problems.append(f"run 2: {exc}")
    frozen_ml = cfg.backtest.frozen_ml_config_hash
    if frozen_ml is None or ml_config_hash(cfg) != frozen_ml:
        problems.append(f"the ML config hash {ml_config_hash(cfg)} is not the frozen {frozen_ml}")
    point: PointBatch | None = None
    try:
        point = holdout_point_batch(engine, region_id, str(frozen_ml))
    except LookupError as exc:
        problems.append(str(exc))
    dev_qr: QrSource | None = None
    try:
        dev_qr = check_qr_batch(engine, cfg)
    except (LookupError, ValueError) as exc:
        problems.append(f"development QR batch: {exc}")
    qr_hold: QrHoldoutBatch | None = None
    if point is not None:
        try:
            qr_hold = qr_holdout_batch(engine, cfg, region_id, point.batch_id)
        except (LookupError, ValueError) as exc:
            problems.append(str(exc))
    dev_runs: dict[str, IntervalRun] = {}
    dev_halfhour: dict[str, IntervalRun] = {}
    try:
        dev_runs = find_runs(engine, cfg, region_id)
        dev_halfhour = find_halfhour_runs(engine, cfg, region_id)
    except (LookupError, ValueError) as exc:
        problems.append(f"development interval runs: {exc}")
    if stored := stored_holdout_batches(engine, region_id, freeze.config_hash):
        problems.append(
            f"the holdout intervals are already scored for interval recipe "
            f"{freeze.config_hash} (batches {', '.join(stored)}); U5 allows one run"
        )
    if problems:
        raise HoldoutIntervalsRefused("; ".join(problems))
    assert window is not None and point is not None and dev_qr is not None
    assert qr_hold is not None
    return HoldoutInputs(window, freeze, point, dev_qr, qr_hold, dev_runs, dev_halfhour)


# --------------------------------------------------------------------------- joined inputs


@dataclass(frozen=True, slots=True)
class Sources:
    """Where the joined inputs come from (small, so it pickles cheaply to the workers)."""

    cfg: AppConfig
    region_id: str
    backtest_dir: Path
    dev_batch: str
    holdout_batch: str
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    dev_qr_dir: Path
    holdout_qr_dir: Path


def joined_batch(src: Sources) -> StoredBatch:
    """Run 2's batch before the holdout followed by ``holdout_main``'s, checked contiguous."""
    dev = read_stored_batch(src.backtest_dir, src.dev_batch, TARGET).restrict(end=src.holdout_start)
    hold = read_stored_batch(src.backtest_dir, src.holdout_batch, TARGET).restrict(
        start=src.holdout_start, end=src.holdout_end
    )
    if set(dev.forecasts) != set(hold.forecasts):
        raise ValueError(
            f"run 2 holds {sorted(dev.forecasts)}, holdout_main {sorted(hold.forecasts)}: "
            "the common mask would differ across the boundary"
        )
    if dev.n_leads != hold.n_leads:
        raise ValueError("run 2 and holdout_main have different leads")
    times = pd.DatetimeIndex(dev.origin_times.append(hold.origin_times))
    step = pd.Timedelta(minutes=5 * ForecastSetup.from_config(src.cfg).issue_every_steps)
    if len(hold.origin_times) == 0 or not bool((times[1:] - times[:-1] == step).all()):
        raise ValueError("the joined issue times are not one contiguous grid")
    return StoredBatch(
        f"{dev.batch_id}+{hold.batch_id}",
        times,
        np.concatenate([dev.actual, hold.actual]),
        {m: np.concatenate([dev.forecasts[m], hold.forecasts[m]]) for m in dev.forecasts},
    )


def joined_quantiles(
    src: Sources, keys: Sequence[str], times: pd.DatetimeIndex
) -> dict[str, np.ndarray]:
    """Step 2's quantiles before the holdout followed by step 6a's, on ``times``."""
    dev_t, dev_q = read_qr_quantiles(src.dev_qr_dir, keys)
    hold_t, hold_q = read_qr_quantiles(src.holdout_qr_dir, keys)
    d = np.asarray(dev_t < src.holdout_start)
    h = np.asarray((hold_t >= src.holdout_start) & (hold_t < src.holdout_end))
    if not same_instants(pd.DatetimeIndex(dev_t[d].append(hold_t[h])), times):
        raise ValueError("the joined QR quantiles' issue times differ from the point inputs'")
    return {k: np.concatenate([dev_q[k][d], hold_q[k][h]]) for k in keys}


def check_qr_overlap(src: Sources, keys: Sequence[str]) -> int:
    """Step 6a's quantiles of development issue times must equal step 2's; returns the count."""
    dev_t, dev_q = read_qr_quantiles(src.dev_qr_dir, keys)
    hold_t, hold_q = read_qr_quantiles(src.holdout_qr_dir, keys)
    h = np.asarray(hold_t < src.holdout_start)
    if not h.any():
        raise HoldoutIntervalsReproductionFailed("step 6a's batch has no development issue time")
    d = np.asarray((dev_t >= hold_t[h][0]) & (dev_t <= hold_t[h][-1]))
    if not same_instants(dev_t[d], hold_t[h]):
        raise HoldoutIntervalsReproductionFailed("step 6a's development issue times differ")
    for k in keys:
        if not np.array_equal(dev_q[k][d], hold_q[k][h], equal_nan=True):
            raise HoldoutIntervalsReproductionFailed(
                f"step 6a's {k} quantiles of development issue times differ from step 2's"
            )
    return int(h.sum())


def _bands(cfg: AppConfig) -> list[tuple[int, int]]:
    return [(int(b[0]), int(b[1])) for b in cfg.ml.lead_bands]


# --------------------------------------------------------------------------- the workers


@dataclass(frozen=True, slots=True)
class Task:
    """One worker's job: 5-minute conformal or CQR on the joined inputs, or a half-hour run."""

    kind: str  # "conformal", "cqr" or "period"
    src: Sources
    model: str | None = None
    level: int | None = None
    period: PeriodTask | None = None


def _conformal(task: Task) -> ConformalResult:
    """:func:`~gridcast.intervals.batch.conformal_for` on the joined point inputs."""
    cfg, src = task.src.cfg, task.src
    assert task.model is not None
    batch = joined_batch(src)
    setup = ForecastSetup.from_config(cfg)
    groups = interval_groups(cfg, src.region_id, batch)
    mask = batch.common_mask()
    problem = point_problem(
        batch.origin_times,
        setup.issue_every_steps,
        setup.latency_steps,
        batch.actual,
        batch.forecasts[task.model],
        mask,
        groups,
        cfg.intervals.levels,
    )
    del batch
    gc.collect()
    log.info("intervals.holdout.conformal.start", model=task.model)
    result = run_conformal(
        problem,
        candidate_grid(cfg),
        cfg.intervals.levels,
        selection_retrains(cfg, src.holdout_end),
        _bands(cfg),
        cfg.intervals.selection_weeks,
    )
    log.info("intervals.holdout.conformal.done", model=task.model)
    return result


def _cqr(task: Task) -> ConformalResult:
    """:func:`~gridcast.intervals.cqr_batch.cqr_for` on the joined point and QR inputs."""
    cfg, src = task.src.cfg, task.src
    assert task.level is not None
    batch = joined_batch(src)
    lo_key, hi_key = interval_quantiles(task.level)
    q = joined_quantiles(src, [lo_key, hi_key], batch.origin_times)
    setup = ForecastSetup.from_config(cfg)
    problem = quantile_problem(
        batch.origin_times,
        setup.issue_every_steps,
        setup.latency_steps,
        batch.actual,
        {task.level: (q[lo_key], q[hi_key])},
        batch.common_mask(),
        interval_groups(cfg, src.region_id, batch),
    )
    del batch, q
    gc.collect()
    log.info("intervals.holdout.cqr.start", level=task.level)
    result = run_conformal(
        problem,
        candidate_grid(cfg),
        [task.level],
        selection_retrains(cfg, src.holdout_end),
        _bands(cfg),
        cfg.intervals.selection_weeks,
    )
    log.info("intervals.holdout.cqr.done", level=task.level)
    return result


def run_task(task: Task) -> ConformalResult:
    if task.kind == "conformal":
        return _conformal(task)
    if task.kind == "cqr":
        return _cqr(task)
    if task.kind == "period" and task.period is not None:
        return period_run(task.period)
    raise ValueError(f"unknown task {task.kind!r}")


# --------------------------------------------------------------------------- reproduction


@dataclass(frozen=True, slots=True)
class DevReproduction:
    run: str
    run_id: int
    batch_id: str
    issue_times: int
    bounds_equal: bool
    selections_equal: bool
    selections: int
    differing_bounds: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.bounds_equal and self.selections_equal


def compare_development(
    name: str,
    run: IntervalRun,
    result: ConformalResult,
    holdout_start: pd.Timestamp,
    data_root: Path,
) -> DevReproduction:
    """The development part of a continuous result against the stored development run."""
    rows = np.asarray(result.origin_times < holdout_start)
    times = result.origin_times[rows]
    directory = data_root / run.directory
    differing = []
    for level in result.lower:
        for side, values in (("lo", result.lower[level]), ("hi", result.upper[level])):
            file = f"{name}__{side}{level}.parquet"
            t, stored = read_wide(directory / file)
            if not same_instants(t, times) or not np.array_equal(
                stored, values[rows].astype(np.float64), equal_nan=True
            ):
                differing.append(file)
    stored_sel = [tuple(s[f] for f in SELECTION_FIELDS) for s in run.config.get("selections", [])]
    new_sel = [
        tuple(asdict(s)[f] for f in SELECTION_FIELDS)
        for s in result.selections
        if pd.Timestamp(s.retrain) < holdout_start
    ]
    return DevReproduction(
        name,
        run.run_id,
        run.batch_id,
        len(times),
        not differing,
        stored_sel == new_sel,
        len(new_sel),
        tuple(differing),
    )


# --------------------------------------------------------------------------- scoring


def holdout_part(result: ConformalResult, start: pd.Timestamp) -> ConformalResult:
    """The result's bounds for issue times from ``start`` (selections and diagnostics kept)."""
    rows = np.asarray(result.origin_times >= start)
    return ConformalResult(
        origin_times=result.origin_times[rows],
        lower={lv: v[rows] for lv, v in result.lower.items()},
        upper={lv: v[rows] for lv, v in result.upper.items()},
        candidates=result.candidates,
        selections=result.selections,
        daily=result.daily,
        clipped=result.clipped,
        crossed=result.crossed,
    )


def raw_qr_metrics(
    quantiles: Mapping[str, np.ndarray],
    crossed: np.ndarray,
    taus: Sequence[float],
    levels: Sequence[int],
    actual: np.ndarray,
    mask: np.ndarray,
    point: np.ndarray,
) -> pd.DataFrame:
    """Step 2's raw-QR metrics (:func:`~gridcast.intervals.qr_batch.qr_metrics`) of stored
    rearranged quantiles and crossing flags."""
    if any(np.any(mask & ~np.isfinite(v)) for v in quantiles.values()):
        raise ValueError("a scored pair has no quantile forecast")
    lower, upper = {}, {}
    for level in levels:
        lo, hi = interval_quantiles(level)
        lower[level], upper[level] = quantiles[lo], quantiles[hi]
    out = [bounds_metrics(lower, upper, actual, mask, point)]
    extra: list[tuple[str, int, float, int]] = []
    for tau in taus:
        key = quantile_key(tau)
        values = quantiles[key].astype(np.float64)
        with np.errstate(invalid="ignore"):
            extra += per_lead_rows(f"pinball_{key}", pinball(values, actual, tau), mask)
            extra += per_lead_rows(f"share_le_{key}", (actual <= values).astype(float), mask)
    extra += per_lead_rows("crossing", crossed.astype(float), mask)
    out.append(pd.DataFrame(extra, columns=["metric", "horizon", "value", "n"]))
    return pd.concat(out, ignore_index=True)


def zero_width(result: ConformalResult) -> dict[str, int]:
    """Per level, intervals of zero width (crossed intervals are output at their midpoint)."""
    out = {}
    for level in result.lower:
        lo, hi = result.lower[level], result.upper[level]
        out[str(level)] = int((np.isfinite(lo) & (lo == hi)).sum())
    return out


@dataclass(slots=True)
class ScoredRun:
    name: str
    resolution: str  # RES_5 or RES_30
    method: str  # conformal, cqr, qr
    point_model: str
    result: ConformalResult | None  # holdout part (None for raw QR)
    metrics: pd.DataFrame
    scored_pairs: int
    total_pairs: int


@dataclass(slots=True)
class HoldoutIntervalsRun:
    inputs: HoldoutInputs
    times: pd.DatetimeIndex  # holdout issue times
    runs: list[ScoredRun]
    reproductions: list[DevReproduction]
    qr_overlap_issue_times: int
    actual_5: np.ndarray  # stored float32 actuals of the holdout issue times
    actual_30: np.ndarray  # half-hour period means (float64)
    retrains: list[str]
    seconds: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)


def run_holdout_intervals(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    inputs: HoldoutInputs,
    workers: int = 3,
) -> HoldoutIntervalsRun:
    """Build every interval over the continuous run, reproduce, then score the holdout."""
    t_start = time.perf_counter()
    window = inputs.window
    start, end = window.holdout_start, window.holdout_end
    src = Sources(
        cfg,
        region_id,
        cfg.data_path(cfg.paths.backtest_dir),
        cfg.intervals.source_batch,
        inputs.holdout_point.batch_id,
        start,
        end,
        inputs.dev_qr.directory,
        inputs.qr_holdout.directory,
    )
    levels = list(cfg.intervals.levels)
    models = list(cfg.intervals.point_models)
    taus = quantile_levels(levels)
    keys = [quantile_key(t) for t in taus]
    overlap = check_qr_overlap(src, keys)

    # Half-hour inputs (period means) from the joined arrays; the large arrays are freed
    # before the workers start.
    joined = joined_batch(src)
    period = build_inputs(
        cfg,
        region_id,
        joined.actual,
        joined.common_mask(),
        joined.origin_times,
        {m: joined.forecasts[m] for m in models},
        joined_quantiles(src, keys, joined.origin_times),
    )
    del joined
    gc.collect()

    tasks = [Task("conformal", src, model=m) for m in models]
    tasks += [Task("cqr", src, level=lv) for lv in levels]
    tasks += [
        Task(
            "period", src, period=PeriodTask(cfg, period.only(model=m), CONFORMAL, model=m, end=end)
        )
        for m in models
    ]
    tasks += [
        Task("period", src, period=PeriodTask(cfg, period.only(level=lv), CQR, level=lv, end=end))
        for lv in levels
    ]
    log.info("intervals.holdout.start", tasks=len(tasks), workers=workers)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        parts = list(pool.map(run_task, tasks))
    n_m, n_l = len(models), len(levels)
    five: dict[str, ConformalResult] = {
        conformal_run_name(m): parts[i] for i, m in enumerate(models)
    }
    five[CQR_RUN_NAME] = merge_levels(parts[n_m : n_m + n_l], levels)
    thirty: dict[str, ConformalResult] = {
        conformal_name(m): parts[n_m + n_l + i] for i, m in enumerate(models)
    }
    thirty[CQR_NAME] = merge_levels(parts[2 * n_m + n_l :], levels)

    # Reproduction of every stored development output, before anything is scored.
    data_root = cfg.data_root()
    reps = [
        compare_development(name, inputs.dev_runs[name], res, start, data_root)
        for name, res in five.items()
    ]
    reps += [
        compare_development(name, inputs.dev_halfhour[name], res, start, data_root)
        for name, res in thirty.items()
    ]
    if failed := [r for r in reps if not r.passed]:
        raise HoldoutIntervalsReproductionFailed(
            "the continuous run does not reproduce the stored development intervals; nothing "
            "scored: "
            + "; ".join(
                f"{r.run} (bounds {', '.join(r.differing_bounds) or 'equal'}, selections "
                f"{'equal' if r.selections_equal else 'differ'})"
                for r in failed
            )
        )
    log.info("intervals.holdout.reproduced", runs=len(reps))

    # Scoring: holdout issue times, holdout_main's stored actuals and common mask.
    hold = read_stored_batch(src.backtest_dir, src.holdout_batch, TARGET).restrict(start, end)
    times = hold.origin_times
    mask = hold.common_mask()
    scored: list[ScoredRun] = []
    for name, res in five.items():
        part = holdout_part(res, start)
        if not same_instants(part.origin_times, times):
            raise ValueError(f"{name}: holdout issue times differ from holdout_main's")
        if name == CQR_RUN_NAME:
            point_model = POINT_MODEL
            metrics = cqr_metrics(part, hold.actual, mask, hold.forecasts[point_model])
            method = "cqr"
        else:
            point_model = name.removeprefix("conformal_")
            metrics = bounds_metrics(
                part.lower, part.upper, hold.actual, mask, hold.forecasts[point_model]
            )
            method = "conformal"
        scored.append(
            ScoredRun(name, RES_5, method, point_model, part, metrics, int(mask.sum()), mask.size)
        )

    q_times, q_hold = read_qr_quantiles(src.holdout_qr_dir, keys)
    c_times, crossed = read_wide(src.holdout_qr_dir / f"{QR_RUN_NAME}__{CROSSED}.parquet")
    q_rows = np.asarray(q_times >= start)
    if not same_instants(q_times[q_rows], times) or not same_instants(c_times, q_times):
        raise ValueError("step 6a's issue times differ from holdout_main's")
    qr_metrics = raw_qr_metrics(
        {k: v[q_rows] for k, v in q_hold.items()},
        crossed[q_rows] > 0,
        taus,
        levels,
        hold.actual,
        mask,
        hold.forecasts[POINT_MODEL],
    )
    scored.append(
        ScoredRun(
            QR_RUN_NAME, RES_5, "qr", POINT_MODEL, None, qr_metrics, int(mask.sum()), mask.size
        )
    )

    p_rows = np.asarray(period.origin_times >= start)
    if not same_instants(period.origin_times[p_rows], times):
        raise ValueError("the half-hour inputs' holdout issue times differ from holdout_main's")
    p_actual, p_mask = period.actual[p_rows], period.mask[p_rows]
    for name, res in thirty.items():
        part = holdout_part(res, start)
        if name == CQR_NAME:
            point_model = CQR_POINT
            metrics = cqr_metrics(part, p_actual, p_mask, period.points[point_model][p_rows])
            method = CQR
        else:
            point_model = name.removeprefix(f"{CONFORMAL}_")
            metrics = bounds_metrics(
                part.lower, part.upper, p_actual, p_mask, period.points[point_model][p_rows]
            )
            method = CONFORMAL
        scored.append(
            ScoredRun(
                name, RES_30, method, point_model, part, metrics, int(p_mask.sum()), p_mask.size
            )
        )

    retrains = [str(r) for r in selection_retrains(cfg, end)]
    return HoldoutIntervalsRun(
        inputs,
        times,
        scored,
        reps,
        overlap,
        hold.actual.astype(np.float32),
        p_actual,
        retrains,
        seconds=round(time.perf_counter() - t_start, 1),
    )


# --------------------------------------------------------------------------- storage


def _insert(
    conn: Connection,
    batch_id: uuid.UUID,
    name: str,
    region_id: str,
    eval_window: tuple[pd.Timestamp, pd.Timestamp],
    holdout: pd.Timestamp,
    config: dict[str, Any],
    sha: str | None,
    artifact: Path,
    metrics: pd.DataFrame,
) -> int:
    """One ``model_runs`` row of kind ``interval`` and its metrics, split ``holdout``."""
    run_id = conn.execute(
        text(
            """
            INSERT INTO model_runs
                (batch_id, model_name, model_kind, target, region_id, eval_start,
                 eval_end, holdout_start, config, git_sha, artifact_path, status,
                 finished_at)
            VALUES (:batch, :name, 'interval', :target, :region, :eval_start,
                    :eval_end, :holdout, CAST(:config AS jsonb), :sha, :artifact,
                    'success', now())
            RETURNING run_id
            """
        ),
        {
            "batch": batch_id,
            "name": name,
            "target": TARGET,
            "region": region_id,
            "eval_start": eval_window[0].to_pydatetime(),
            "eval_end": eval_window[1].to_pydatetime(),
            "holdout": holdout.to_pydatetime(),
            "config": json.dumps(config, default=str),
            "sha": sha,
            "artifact": str(artifact),
        },
    ).scalar_one()
    rows = [
        {
            "run_id": int(run_id),
            "split": SPLIT,
            "segment": "all",
            "horizon": int(r["horizon"]),
            "metric": str(r["metric"]),
            "value": float(r["value"]) if np.isfinite(float(r["value"])) else None,
            "n": int(r["n"]),
        }
        for r in metrics.to_dict(orient="records")
    ]
    conn.execute(
        text(
            """
            INSERT INTO forecast_metrics (run_id, split, segment, horizon, metric, value, n)
            VALUES (:run_id, :split, :segment, :horizon, :metric, :value, :n)
            """
        ),
        rows,
    )
    return int(run_id)


def store_holdout_intervals(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    run: HoldoutIntervalsRun,
    sha: str | None,
) -> tuple[uuid.UUID, dict[str, int]]:
    """Write the holdout bounds and runs as one batch, re-checking the U5 lock first."""
    inputs = run.inputs
    if stored := stored_holdout_batches(engine, region_id, inputs.freeze.config_hash):
        raise HoldoutIntervalsRefused(
            f"holdout interval batches appeared during the run: {', '.join(stored)}"
        )
    batch_id = uuid.uuid4()
    root = cfg.data_path(cfg.intervals.output_dir) / str(batch_id)
    dirs = {RES_5: root / RES_5, RES_30: root / RES_30}
    for res_key, actual, dtype in (
        (RES_5, run.actual_5, np.float32),
        (RES_30, run.actual_30, np.float64),
    ):
        results = {
            r.name: r.result for r in run.runs if r.resolution == res_key and r.result is not None
        }
        save_results(
            dirs[res_key], results, actual, run.times, name_of=lambda n: n, actual_dtype=dtype
        )
    window = inputs.window
    holdout_retrains = [
        r for i, r in enumerate(run.retrains)
        if pd.Timestamp(r) < window.holdout_end
        and (i + 1 == len(run.retrains) or pd.Timestamp(run.retrains[i + 1]) > window.holdout_start)
    ]  # fmt: skip
    reproduction = {
        "passed": True,  # a mismatch raises before scoring
        "development_runs": [asdict(r) for r in run.reproductions],
        "qr_overlap_issue_times": run.qr_overlap_issue_times,
    }
    common: dict[str, Any] = {
        "evaluation_set": EVALUATION_SET,
        "period": "holdout",
        "source_batch": cfg.intervals.source_batch,
        "holdout_point_batch": inputs.holdout_point.batch_id,
        "qr_batch": inputs.dev_qr.batch_id,
        "qr_holdout_batch": inputs.qr_holdout.batch_id,
        "qr_holdout_run_id": inputs.qr_holdout.run_id,
        "interval_config_hash": inputs.freeze.config_hash,
        "qr_config_hash": inputs.freeze.qr_config_hash,
        "frozen_commit": inputs.freeze.frozen_commit,
        "code_sha": inputs.freeze.sha,
        "intervals": cfg.intervals.model_dump(mode="json"),
        "lead_bands": [list(b) for b in cfg.ml.lead_bands],
        "retrains": run.retrains,
        "holdout_retrains": holdout_retrains,
        "calibration_first_issue": str(nem_midnight_utc(str(cfg.ml.first_issue))),
        "holdout_start": str(window.holdout_start),
        "holdout_end": str(window.holdout_end),
        "first_issue": str(run.times[0]),
        "last_issue": str(run.times[-1]),
        "n_origins": len(run.times),
        "complete_window": True,
        "reproduction": reproduction,
        "seconds": run.seconds,
    }
    run_ids: dict[str, int] = {}
    with engine.begin() as conn:
        for r in run.runs:
            config: dict[str, Any] = {
                **common,
                "method": r.method,
                "point_model": r.point_model,
                "resolution_minutes": PERIOD_MINUTES if r.resolution == RES_30 else 5,
                "scored_pairs": r.scored_pairs,
                "total_pairs": r.total_pairs,
            }
            if r.resolution == RES_30:
                config["period_bands"] = [list(b) for b in period_bands(cfg.ml.lead_bands)]
            if r.result is not None:
                # Selections, candidates, clipped and crossed counts of the continuous run
                # (development and holdout issue times); zero-width = holdout part only.
                config |= result_config(r.result)
                config["counts_cover"] = "the continuous run from intervals.first_issue"
                config["zero_width_holdout"] = zero_width(r.result)
                artifact = dirs[r.resolution]
            else:
                artifact = inputs.qr_holdout.directory
            run_ids[r.name] = _insert(
                conn,
                batch_id,
                r.name,
                region_id,
                (run.times[0], run.times[-1]),
                window.holdout_start,
                config,
                sha,
                artifact.relative_to(cfg.data_root()),
                r.metrics,
            )
    return batch_id, run_ids
