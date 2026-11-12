"""Conformal intervals around stored point forecasts, as one stored batch (M3 step 1).

``gridcast intervals-conformal`` reads M2 run 2's development forecasts
(``intervals.source_batch``) for issue times before ``backtest.holdout_start``. For each
model in ``intervals.point_models`` it runs every candidate stream with the per-retrain
selection (:mod:`gridcast.intervals.conformal`, m3_plan.md §2.2). Retrains follow M2's
quarterly schedule; selections (and so the output) start at ``intervals.first_issue``.

Output, under ``<data root>/<intervals.output_dir>/<batch id>/``:

* ``<run name>__lo<L>.parquet`` and ``__hi<L>.parquet``: wide bounds per level
  (``origin_time`` plus ``h001…hHHH``, float32), from ``intervals.first_issue``;
* ``<run name>__daily.parquet``: every candidate's daily statistics (NEM day of issue,
  lead band, level), the material of the per-retrain selection;
* ``actual.parquet``: the actuals of the same issue times, for self-contained scoring.

Database: one ``model_runs`` row per point model (kind ``interval``, name
``conformal_<model>``) whose config records the source batch, the interval config and its
hash, and every selection; ``forecast_metrics`` holds coverage, share below and above,
width and interval score per level and lead (0 = pooled), plus the weighted interval
score, for split ``dev``, segment ``all``. Every metric is computed from the stored
(float32) bounds.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Connection, Engine, text

from gridcast.config import AppConfig
from gridcast.forecasting.origins import ForecastSetup, nem_midnight_utc
from gridcast.forecasting.walkforward import retrain_schedule
from gridcast.intervals.conformal import (
    CandidateGrid,
    ConformalResult,
    point_problem,
    run_conformal,
    tod_labels,
)
from gridcast.intervals.inputs import StoredBatch, read_stored_batch
from gridcast.intervals.scores import SUMMARY_STATS, summarise, weighted_interval_score
from gridcast.logs import get_logger
from gridcast.provenance import git_sha

log = get_logger(__name__)

TARGET = "demand"
METHOD = "conformal"
EVALUATION_SET = "intervals_dev"


def _metric_row(run_id: int, r: dict[Any, Any]) -> dict[str, Any]:
    value = float(r["value"])
    return {
        "run_id": run_id,
        "split": str(r["split"]),
        "segment": str(r["segment"]),
        "horizon": int(r["horizon"]),
        "metric": str(r["metric"]),
        "value": value if np.isfinite(value) else None,
        "n": int(r["n"]),
    }


# Input pointers, not recipe settings: ``qr_batch`` names the step-2 quantiles that CQR reads
# (its own recipe is recorded as ``qr_config_hash``), so adding it left step 1's hash as it was.
# The freeze record (step 5) describes the recipe and is not part of it.
HASH_EXCLUDED: frozenset[str] = frozenset(
    {"output_dir", "qr_batch", "frozen_config_hash", "frozen_qr_config_hash", "frozen_commit"}
)


def interval_config_hash(cfg: AppConfig) -> str:
    """Hash of every setting that changes the intervals (not paths, input pointers or reporting)."""
    payload = {
        "intervals": cfg.intervals.model_dump(mode="json", exclude=set(HASH_EXCLUDED)),
        "forecast": cfg.forecast.model_dump(mode="json"),
        "schedule": {
            "first_issue": str(cfg.ml.first_issue),
            "retrain_every_months": cfg.ml.retrain_every_months,
            "lead_bands": [list(b) for b in cfg.ml.lead_bands],
        },
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def candidate_grid(cfg: AppConfig) -> CandidateGrid:
    g = cfg.intervals.grid
    return CandidateGrid(
        tuple(float(w) for w in g.window_weeks), tuple(g.groupings), tuple(g.aci_gammas)
    )


def selection_retrains(cfg: AppConfig, end: pd.Timestamp) -> list[pd.Timestamp]:
    """M2's quarterly retrains from ``intervals.first_issue`` (inclusive) to ``end``."""
    first = nem_midnight_utc(str(cfg.intervals.first_issue))
    schedule = retrain_schedule(cfg.ml.first_issue, end, cfg.ml.retrain_every_months)
    out = [r for r in schedule if r >= first]
    if not out or out[0] != first:
        raise ValueError("intervals.first_issue must be one of M2's retrain dates")
    return out


def run_name(model: str) -> str:
    return f"{METHOD}_{model}"


@dataclass(frozen=True, slots=True)
class PointTask:
    """Everything one worker needs to build one point model's intervals."""

    cfg: AppConfig
    region_id: str
    directory: Path
    batch_id: str
    model: str
    end: pd.Timestamp


def _load(task: PointTask) -> StoredBatch:
    batch = read_stored_batch(task.directory, task.batch_id, TARGET)
    return batch.restrict(end=task.end)


def interval_groups(
    cfg: AppConfig, region_id: str, batch: StoredBatch
) -> dict[str, tuple[np.ndarray, int]]:
    """Group labels of every pair's target per grouping (``lead``: one group)."""
    setup = ForecastSetup.from_config(cfg)
    if batch.n_leads != setup.horizon_steps:
        raise ValueError("the stored batch does not have forecast.horizon_steps leads")
    return {
        "lead": (np.zeros(batch.actual.shape, dtype=np.int8), 1),
        "lead_tod": tod_labels(
            batch.origin_times,
            batch.n_leads,
            setup.issue_every_steps,
            cfg.region(region_id).local_tz,
            cfg.intervals.tod_block_starts,
        ),
    }


def conformal_for(task: PointTask) -> ConformalResult:
    """Every candidate stream and the selection for one point model (a worker's job)."""
    cfg = task.cfg
    batch = _load(task)
    if task.model not in batch.forecasts:
        raise KeyError(f"batch {task.batch_id} has no forecasts from {task.model!r}")
    setup = ForecastSetup.from_config(cfg)
    groups = interval_groups(cfg, task.region_id, batch)
    problem = point_problem(
        batch.origin_times,
        setup.issue_every_steps,
        setup.latency_steps,
        batch.actual,
        batch.forecasts[task.model],
        batch.common_mask(),
        groups,
        cfg.intervals.levels,
    )
    log.info("intervals.conformal.start", model=task.model, issue_times=len(batch.origin_times))
    result = run_conformal(
        problem,
        candidate_grid(cfg),
        cfg.intervals.levels,
        selection_retrains(cfg, task.end),
        [(int(b[0]), int(b[1])) for b in cfg.ml.lead_bands],
        cfg.intervals.selection_weeks,
    )
    log.info("intervals.conformal.done", model=task.model)
    return result


def bounds_metrics(
    lower: Mapping[int, np.ndarray],
    upper: Mapping[int, np.ndarray],
    actual: np.ndarray,
    mask: np.ndarray,
    point: np.ndarray,
) -> pd.DataFrame:
    """Long table (metric, horizon, value, n) of bounds keyed by level; horizon 0 = pooled.

    Coverage, share below and above, width and interval score per level, plus the weighted
    interval score with ``point`` as the median. Bounds are scored as float64 copies of
    whatever is passed (the stored float32 values).
    """
    rows: list[tuple[str, int, float, int]] = []
    intervals = {}
    for level in lower:
        lo = lower[level].astype(np.float64)
        hi = upper[level].astype(np.float64)
        intervals[level] = (lo, hi)
        pooled, n = summarise(lo, hi, actual, mask, level)
        per, n_h = summarise(lo, hi, actual, mask, level, axis=0)
        for stat in SUMMARY_STATS:
            rows.append((f"{stat}_{level}", 0, float(pooled[stat]), int(n)))
            rows.extend(
                (f"{stat}_{level}", h + 1, float(per[stat][h]), int(n_h[h]))
                for h in range(actual.shape[1])
            )
    wis = np.where(mask, weighted_interval_score(point, intervals, actual), 0.0)
    rows.extend(per_lead_rows("wis", wis, mask))
    return pd.DataFrame(rows, columns=["metric", "horizon", "value", "n"])


def per_lead_rows(
    metric: str, values: np.ndarray, mask: np.ndarray
) -> list[tuple[str, int, float, int]]:
    """Pooled (horizon 0) and per-lead means of ``values`` over ``mask``."""
    v = np.where(mask, values, 0.0)
    n_h = mask.sum(axis=0)
    n = int(mask.sum())
    rows = [(metric, 0, float(v.sum() / n) if n else float("nan"), n)]
    rows.extend(
        (metric, h + 1, float(v[:, h].sum() / n_h[h]) if n_h[h] else float("nan"), int(n_h[h]))
        for h in range(values.shape[1])
    )
    return rows


def interval_metrics(
    result: ConformalResult, actual: np.ndarray, mask: np.ndarray, point: np.ndarray
) -> pd.DataFrame:
    """:func:`bounds_metrics` of a conformal result's stored (float32) bounds."""
    return bounds_metrics(result.lower, result.upper, actual, mask, point)


def wide(times: pd.DatetimeIndex, values: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=[f"h{h:03d}" for h in range(1, values.shape[1] + 1)])
    frame.insert(0, "origin_time", times)
    return frame


def save_results(
    directory: Path,
    results: dict[str, ConformalResult],
    actual: np.ndarray,
    times: pd.DatetimeIndex,
    name_of: Callable[[str], str] = run_name,
    actual_dtype: type[np.floating[Any]] = np.float32,
) -> None:
    """Bounds per level and side, daily candidate statistics and the actuals (wide parquet).

    The actuals are stored as float32 (M2's stored values), or as float64 where they are
    means the run computed (half-hour periods), so that they rescore exactly.
    """
    directory.mkdir(parents=True, exist_ok=False)
    wide(times, actual.astype(actual_dtype)).to_parquet(
        directory / "actual.parquet", index=False, compression="zstd"
    )
    for model, res in results.items():
        if not res.origin_times.equals(times):
            raise ValueError(f"{model}: output issue times differ")
        name = name_of(model)
        for level in res.lower:
            for side, values in (("lo", res.lower[level]), ("hi", res.upper[level])):
                wide(times, values).to_parquet(
                    directory / f"{name}__{side}{level}.parquet", index=False, compression="zstd"
                )
        res.daily.to_parquet(directory / f"{name}__daily.parquet", index=False)


def insert_run(
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
    """One ``model_runs`` row of kind ``interval``, plus its metrics (split ``dev``, all pairs)."""
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
    records = metrics.assign(split="dev", segment="all").to_dict(orient="records")
    conn.execute(
        text(
            """
            INSERT INTO forecast_metrics (run_id, split, segment, horizon, metric, value, n)
            VALUES (:run_id, :split, :segment, :horizon, :metric, :value, :n)
            """
        ),
        [_metric_row(int(run_id), r) for r in records],
    )
    return int(run_id)


def result_config(res: ConformalResult) -> dict[str, Any]:
    """The candidates, every selection, and the crossed and clipped counts of a result."""
    return {
        "candidates": [c.key for c in res.candidates],
        "selections": [asdict(s) for s in res.selections],
        "crossed": {str(k): v for k, v in res.crossed.items()},
        "clipped": res.clipped.to_dict(orient="records"),
    }


def store_runs(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    batch_id: uuid.UUID,
    results: dict[str, ConformalResult],
    metrics: dict[str, pd.DataFrame],
    artifact: Path,
    run_config: dict[str, Any],
    sha: str | None = None,
) -> dict[str, int]:
    """One ``model_runs`` row (kind ``interval``) per point model, plus its metrics."""
    sha = sha or git_sha()
    holdout = nem_midnight_utc(str(cfg.backtest.holdout_start))
    run_ids: dict[str, int] = {}
    with engine.begin() as conn:
        for model, res in results.items():
            config = {
                **run_config,
                "method": METHOD,
                "point_model": model,
                **result_config(res),
            }
            run_ids[model] = insert_run(
                conn,
                batch_id,
                run_name(model),
                region_id,
                (res.origin_times[0], res.origin_times[-1]),
                holdout,
                config,
                sha,
                artifact,
                metrics[model],
            )
    return run_ids


def check_source_batch(engine: Engine, batch_id: str) -> dict[str, Any]:
    """The source batch must be a complete development ``main`` set; return its config."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT config FROM model_runs
                WHERE batch_id = CAST(:b AS uuid) AND model_name = 'lgbm_C'
                """
            ),
            {"b": batch_id},
        ).one_or_none()
    if row is None:
        raise LookupError(f"source batch {batch_id} has no lgbm_C run")
    config: dict[str, Any] = row[0]
    if config.get("evaluation_set") != "main" or not config.get("complete_window"):
        raise ValueError(f"source batch {batch_id} is not a complete 'main' development set")
    return config


def run_intervals_conformal(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    models: list[str] | None = None,
    workers: int = 4,
) -> tuple[uuid.UUID, dict[str, pd.DataFrame]]:
    """Build, score and store the conformal interval batch; return its id and metrics."""
    source = cfg.intervals.source_batch
    source_config = check_source_batch(engine, source)
    models = models or list(cfg.intervals.point_models)
    end = nem_midnight_utc(str(cfg.backtest.holdout_start))
    directory = cfg.data_path(cfg.paths.backtest_dir)
    tasks = [PointTask(cfg, region_id, directory, source, m, end) for m in models]
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        results = dict(zip(models, pool.map(conformal_for, tasks), strict=True))

    batch = _load(tasks[0])
    times = next(iter(results.values())).origin_times
    rows = np.asarray(batch.origin_times >= times[0])
    actual, mask = batch.actual[rows], batch.common_mask()[rows]
    metrics = {}
    for model, res in results.items():
        point = batch.forecasts[model][rows]
        metrics[model] = interval_metrics(res, actual, mask, point)

    batch_id = uuid.uuid4()
    out_dir = cfg.data_path(cfg.intervals.output_dir) / str(batch_id)
    save_results(out_dir, results, actual, times)
    run_config = {
        "evaluation_set": EVALUATION_SET,
        "source_batch": source,
        "source_config_hash": source_config.get("config_hash"),
        "interval_config_hash": interval_config_hash(cfg),
        "intervals": cfg.intervals.model_dump(mode="json"),
        "lead_bands": [list(b) for b in cfg.ml.lead_bands],
        "retrains": [str(r) for r in selection_retrains(cfg, end)],
        "first_issue": str(times[0]),
        "last_issue": str(times[-1]),
        "n_origins": len(times),
        "scored_pairs": int(mask.sum()),
        "total_pairs": int(mask.size),
    }
    store_runs(
        engine,
        cfg,
        region_id,
        batch_id,
        results,
        metrics,
        out_dir.relative_to(cfg.data_root()),
        run_config,
    )
    return batch_id, metrics
