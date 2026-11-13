"""Conformalised quantile regression on quantile LightGBM C (M3 step 3, ``gridcast intervals-cqr``).

Inputs:

* the QR development batch (``intervals.qr_batch``, step 2): rearranged quantiles for every
  development issue time from ``ml.first_issue``;
* run 2's stored batch (``intervals.source_batch``): the actuals, the common mask (scored and
  calibration pairs, as in step 1) and lgbm_C's point forecast, the median of the WIS.

For each level, the central quantiles (q̂_lo, q̂_hi) are corrected by step 1's rolling
conformal machinery with one score per side, q̂_lo − y and y − q̂_hi (m3_plan.md §2.2). It
uses the same candidate streams (window × grouping × ACI step) and the same per-retrain
selection by mean interval score over the selection slice. Issue times before
``intervals.first_issue`` are calibration warm-up only, as in step 1.

Levels share nothing (their own scores, ACI state and selection), so each level runs in its
own process. :func:`merge_levels` joins the parts into the result of a single run, which a
unit test checks.

Output, under ``<data root>/<intervals.output_dir>/<batch id>/``, laid out as step 1's:
``cqr_lgbm_C__lo<L>.parquet``/``__hi<L>.parquet``, ``cqr_lgbm_C__daily.parquet`` and
``actual.parquet``.

Database: one ``model_runs`` row (kind ``interval``, name ``cqr_lgbm_C``) records the QR
batch and its recipe hash, the interval config hash and every selection. ``forecast_metrics``
holds step 1's metrics, plus, per quantile level of the corrected bounds, the pinball loss
and the share of actuals at or below the bound (the reliability diagram). It also holds the
share of pairs whose levels are not nested.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from gridcast.config import AppConfig
from gridcast.forecasting.origins import ForecastSetup, nem_midnight_utc
from gridcast.intervals.batch import (
    EVALUATION_SET,
    TARGET,
    bounds_metrics,
    candidate_grid,
    check_source_batch,
    insert_run,
    interval_config_hash,
    interval_groups,
    per_lead_rows,
    result_config,
    save_results,
    selection_retrains,
)
from gridcast.intervals.conformal import TAILS, ConformalResult, quantile_problem, run_conformal
from gridcast.intervals.inputs import StoredBatch, read_stored_batch, read_wide
from gridcast.intervals.qr_batch import RUN_NAME as QR_RUN_NAME
from gridcast.intervals.qr_batch import qr_config_hash
from gridcast.intervals.quantile import interval_quantiles, quantile_key, quantile_levels
from gridcast.intervals.scores import pinball
from gridcast.logs import get_logger
from gridcast.provenance import git_sha

log = get_logger(__name__)

METHOD = "cqr"
POINT_MODEL = "lgbm_C"
RUN_NAME = f"{METHOD}_{POINT_MODEL}"


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True, slots=True)
class QrSource:
    """The stored QR batch that CQR corrects."""

    batch_id: str
    run_id: int
    directory: Path
    config: dict[str, Any]


def check_qr_batch(engine: Engine, cfg: AppConfig) -> QrSource:
    """``intervals.qr_batch`` must be a complete, reproduced QR run of the current recipe."""
    batch = cfg.intervals.qr_batch
    if not batch:
        raise ValueError("intervals.qr_batch is not set: run gridcast backtest-qr first")
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT run_id, config, artifact_path FROM model_runs
                WHERE batch_id = CAST(:b AS uuid) AND model_name = :name
                """
            ),
            {"b": batch, "name": QR_RUN_NAME},
        ).one_or_none()
    if row is None:
        raise LookupError(f"QR batch {batch} has no {QR_RUN_NAME} run")
    config: dict[str, Any] = row.config
    if config.get("evaluation_set") != EVALUATION_SET or not config.get("complete_window"):
        raise ValueError(f"QR batch {batch} is not a complete development run")
    if not config.get("point_reproduction", {}).get("passed"):
        raise ValueError(f"QR batch {batch} did not reproduce run 2's point forecasts")
    if config.get("source_batch") != cfg.intervals.source_batch:
        raise ValueError(
            f"QR batch {batch} was trained against source batch {config.get('source_batch')}, "
            f"not intervals.source_batch {cfg.intervals.source_batch}"
        )
    current = qr_config_hash(cfg)
    if config.get("qr_config_hash") != current:
        raise ValueError(
            f"QR batch {batch} has QR config hash {config.get('qr_config_hash')}, the current "
            f"recipe {current}: CQR must correct the quantiles of the same recipe"
        )
    return QrSource(batch, int(row.run_id), cfg.data_root() / row.artifact_path, config)


def read_qr_quantiles(
    directory: Path, keys: Sequence[str]
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Stored (rearranged) quantiles by key, checked to share issue times."""
    times: pd.DatetimeIndex | None = None
    out: dict[str, np.ndarray] = {}
    for key in keys:
        t, values = read_wide(directory / f"{QR_RUN_NAME}__{key}.parquet")
        if times is None:
            times = t
        elif not times.equals(t):
            raise ValueError(f"QR quantile {key}: issue times differ from the other quantiles")
        out[key] = values
    if times is None:
        raise ValueError("no quantiles requested")
    return times, out


def check_alignment(batch: StoredBatch, qr_dir: Path, keys: Sequence[str]) -> None:
    """The QR batch must cover run 2's issue times exactly, with the same actuals, and have
    every quantile wherever run 2's common mask scores or calibrates a pair."""
    times, actual = read_wide(qr_dir / "actual.parquet")
    if not times.equals(batch.origin_times):
        raise ValueError(
            f"the QR batch has {len(times)} issue times ({times[0]} → {times[-1]}); run 2's "
            f"development batch has {len(batch.origin_times)}"
        )
    if not np.array_equal(
        actual.astype(np.float32), batch.actual.astype(np.float32), equal_nan=True
    ):
        raise ValueError("the QR batch's actuals differ from run 2's stored actuals")
    q_times, quantiles = read_qr_quantiles(qr_dir, keys)
    if not q_times.equals(batch.origin_times):
        raise ValueError("the QR quantiles' issue times differ from run 2's")
    mask = batch.common_mask()
    for key, values in quantiles.items():
        missing = int((mask & ~np.isfinite(values)).sum())
        if missing:
            raise ValueError(f"QR quantile {key} is missing on {missing} pairs of the common mask")


# --------------------------------------------------------------------------- the run


@dataclass(frozen=True, slots=True)
class LevelTask:
    """Everything one worker needs to build one level's CQR intervals."""

    cfg: AppConfig
    region_id: str
    backtest_dir: Path
    qr_dir: Path
    level: int
    end: pd.Timestamp


def _source(cfg: AppConfig, directory: Path, end: pd.Timestamp) -> StoredBatch:
    return read_stored_batch(directory, cfg.intervals.source_batch, TARGET).restrict(end=end)


def cqr_for(task: LevelTask) -> ConformalResult:
    """Every candidate stream and the selection of one level (a worker's job)."""
    cfg = task.cfg
    batch = _source(cfg, task.backtest_dir, task.end)
    lo_key, hi_key = interval_quantiles(task.level)
    times, q = read_qr_quantiles(task.qr_dir, [lo_key, hi_key])
    if not times.equals(batch.origin_times):
        raise ValueError("the QR quantiles' issue times differ from run 2's")
    setup = ForecastSetup.from_config(cfg)
    problem = quantile_problem(
        batch.origin_times,
        setup.issue_every_steps,
        setup.latency_steps,
        batch.actual,
        {task.level: (q[lo_key], q[hi_key])},
        batch.common_mask(),
        interval_groups(cfg, task.region_id, batch),
    )
    log.info("intervals.cqr.start", level=task.level, issue_times=len(batch.origin_times))
    result = run_conformal(
        problem,
        candidate_grid(cfg),
        [task.level],
        selection_retrains(cfg, task.end),
        [(int(b[0]), int(b[1])) for b in cfg.ml.lead_bands],
        cfg.intervals.selection_weeks,
    )
    log.info("intervals.cqr.done", level=task.level)
    return result


def merge_levels(parts: Sequence[ConformalResult], levels: Sequence[int]) -> ConformalResult:
    """Join per-level runs into the result one run over ``levels`` gives (same row orders)."""
    if not parts:
        raise ValueError("nothing to merge")
    first = parts[0]
    keys = [c.key for c in first.candidates]
    for part in parts[1:]:
        if not part.origin_times.equals(first.origin_times):
            raise ValueError("the per-level runs have different issue times")
        if [c.key for c in part.candidates] != keys:
            raise ValueError("the per-level runs have different candidates")
    lower = {lv: v for p in parts for lv, v in p.lower.items()}
    upper = {lv: v for p in parts for lv, v in p.upper.items()}
    if sorted(lower) != sorted(levels) or len(lower) != sum(len(p.lower) for p in parts):
        raise ValueError("every level must come from exactly one part")
    level_pos = {lv: i for i, lv in enumerate(levels)}
    cand_pos = {k: i for i, k in enumerate(keys)}
    tail_pos = {t: i for i, t in enumerate(TAILS)}
    # One run loops retrain → band → level; bands are labelled hNNN-NNN, so they sort in order.
    selections = sorted(
        (s for p in parts for s in p.selections),
        key=lambda s: (s.retrain, s.band, level_pos[s.level]),
    )
    daily = pd.concat([p.daily for p in parts], ignore_index=True)
    order = np.lexsort(
        (
            daily["level"].map(level_pos).to_numpy(),
            daily["candidate"].map(cand_pos).to_numpy(),
            daily["band"].to_numpy(),
            daily["nem_day"].to_numpy(),
        )
    )
    clipped = pd.concat([p.clipped for p in parts], ignore_index=True)
    c_order = np.lexsort(
        (
            clipped["tail"].map(tail_pos).to_numpy(),
            clipped["level"].map(level_pos).to_numpy(),
            clipped["candidate"].map(cand_pos).to_numpy(),
        )
    )
    return ConformalResult(
        origin_times=first.origin_times,
        lower={lv: lower[lv] for lv in levels},
        upper={lv: upper[lv] for lv in levels},
        candidates=list(first.candidates),
        selections=selections,
        daily=daily.iloc[order].reset_index(drop=True),
        clipped=clipped.iloc[c_order].reset_index(drop=True),
        crossed={lv: c for p in parts for lv, c in p.crossed.items()},
    )


def cqr_metrics(
    result: ConformalResult, actual: np.ndarray, mask: np.ndarray, point: np.ndarray
) -> pd.DataFrame:
    """Step 1's metrics of the stored (float32) bounds, plus, per quantile level of a bound,
    the pinball loss and the share of actuals at or below it, and the share of pairs whose
    levels are not nested (a wider level's bound inside a narrower one's)."""
    out = [bounds_metrics(result.lower, result.upper, actual, mask, point)]
    extra: list[tuple[str, int, float, int]] = []
    levels = sorted(result.lower)
    for level in levels:
        tau_lo, tau_hi = quantile_levels([level])
        for tau, bound in ((tau_lo, result.lower[level]), (tau_hi, result.upper[level])):
            key = quantile_key(tau)
            values = bound.astype(np.float64)
            with np.errstate(invalid="ignore"):
                extra += per_lead_rows(f"pinball_{key}", pinball(values, actual, tau), mask)
                extra += per_lead_rows(f"share_le_{key}", (actual <= values).astype(float), mask)
    nested_fail = np.zeros(actual.shape, dtype=bool)
    for narrow, wider in pairwise(levels):
        nested_fail |= result.lower[wider] > result.lower[narrow]
        nested_fail |= result.upper[wider] < result.upper[narrow]
    extra += per_lead_rows("non_nested", nested_fail.astype(float), mask)
    out.append(pd.DataFrame(extra, columns=["metric", "horizon", "value", "n"]))
    return pd.concat(out, ignore_index=True)


@dataclass(slots=True)
class CqrRun:
    batch_id: uuid.UUID
    run_id: int
    result: ConformalResult
    metrics: pd.DataFrame
    run_config: dict[str, Any]


def run_intervals_cqr(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    workers: int = 3,
    sha: str | None = None,
) -> CqrRun:
    """Build, score and store the CQR batch of quantile LightGBM C (development)."""
    t_start = time.perf_counter()
    source_config = check_source_batch(engine, cfg.intervals.source_batch)
    qr = check_qr_batch(engine, cfg)
    levels = list(cfg.intervals.levels)
    keys = [quantile_key(t) for t in quantile_levels(levels)]
    end = nem_midnight_utc(str(cfg.backtest.holdout_start))
    backtest_dir = cfg.data_path(cfg.paths.backtest_dir)
    batch = _source(cfg, backtest_dir, end)
    check_alignment(batch, qr.directory, keys)

    tasks = [LevelTask(cfg, region_id, backtest_dir, qr.directory, lv, end) for lv in levels]
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        parts = list(pool.map(cqr_for, tasks))
    result = merge_levels(parts, levels)

    times = result.origin_times
    rows = np.asarray(batch.origin_times >= times[0])
    actual, mask = batch.actual[rows], batch.common_mask()[rows]
    metrics = cqr_metrics(result, actual, mask, batch.forecasts[POINT_MODEL][rows])

    batch_id = uuid.uuid4()
    out_dir = cfg.data_path(cfg.intervals.output_dir) / str(batch_id)
    save_results(out_dir, {POINT_MODEL: result}, actual, times, name_of=lambda _m: RUN_NAME)
    run_config: dict[str, Any] = {
        "evaluation_set": EVALUATION_SET,
        "method": METHOD,
        "point_model": POINT_MODEL,
        "qr_batch": qr.batch_id,
        "qr_run_id": qr.run_id,
        "qr_config_hash": qr.config.get("qr_config_hash"),
        "source_batch": cfg.intervals.source_batch,
        "source_config_hash": source_config.get("config_hash"),
        "interval_config_hash": interval_config_hash(cfg),
        "intervals": cfg.intervals.model_dump(mode="json"),
        "lead_bands": [list(b) for b in cfg.ml.lead_bands],
        "retrains": [str(r) for r in selection_retrains(cfg, end)],
        "calibration_first_issue": str(batch.origin_times[0]),
        "first_issue": str(times[0]),
        "last_issue": str(times[-1]),
        "n_origins": len(times),
        "scored_pairs": int(mask.sum()),
        "total_pairs": int(mask.size),
        **result_config(result),
    }
    run_config["seconds"] = round(time.perf_counter() - t_start, 1)
    holdout = nem_midnight_utc(str(cfg.backtest.holdout_start))
    with engine.begin() as conn:
        run_id = insert_run(
            conn,
            batch_id,
            RUN_NAME,
            region_id,
            (times[0], times[-1]),
            holdout,
            run_config,
            sha or git_sha(),
            out_dir.relative_to(cfg.data_root()),
            metrics,
        )
    return CqrRun(batch_id, run_id, result, metrics, run_config)
