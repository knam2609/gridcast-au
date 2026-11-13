"""Intervals for 30-minute mean demand, for the AEMO comparison (M3 step 5, m3_plan.md §2.6).

AEMO's POE band is published for 30-minute periods. The mean of six 5-minute quantiles is
not a quantile of the six intervals' mean demand, so the comparison uses intervals
recalibrated on half-hour periods: the recipe of steps 1 and 3, applied to period means.

* **Periods.** Period k of issue time o (k = 1 … 48) ends at o + 30k min and holds the
  5-minute leads h = 6k − 5 … 6k. Its target is the mean of their six actuals, as in M2's
  AEMO comparison (:func:`gridcast.evaluation.aemo_benchmark.thirty_minute_means`).
* **Scored pairs.** A period is scored, and calibrates, only if all six of its 5-minute
  pairs are in run 2's common mask.
* **Conformal** (method 1) is built around the period mean of each point model's six
  forecasts; **CQR** (method 3) corrects the period means of the six rearranged QR
  quantiles of each level (``intervals.qr_batch``).
* **The same recipe:** candidate grid, time-of-day blocks (of the period's start), ACI steps
  and per-retrain selection are those of the 5-minute intervals. M2's lead bands map to
  whole periods (5–30 min → period 1, 35 min–2 h → 2–4, 2 h 5 min–6 h → 5–12, the rest
  13–48). Every level is built; the comparison uses the headline level.

Output, under ``<data root>/<intervals.output_dir>/<batch id>/``, laid out as step 1's with
one column per period (``h001…h048``): ``conformal30_<model>__lo<L>.parquet`` and
``__hi<L>.parquet`` for every model of ``intervals.point_models``, the same for
``cqr30_lgbm_C``, each run's ``__daily.parquet``, and ``actual.parquet`` (period means,
float64, the values the run scored).

Database: one ``model_runs`` row (kind ``interval``) per run, recording
``resolution_minutes`` = 30, the period bands, the interval and QR config hashes and every
selection. ``forecast_metrics`` holds step 1's metrics per period (``horizon`` = k; 0 =
pooled), and for CQR also step 3's pinball, reliability and nesting metrics.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine

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
    result_config,
    save_results,
    selection_retrains,
)
from gridcast.intervals.conformal import (
    ConformalResult,
    point_problem,
    quantile_problem,
    run_conformal,
    tod_labels,
)
from gridcast.intervals.cqr_batch import (
    check_alignment,
    check_qr_batch,
    cqr_metrics,
    merge_levels,
    read_qr_quantiles,
)
from gridcast.intervals.inputs import read_stored_batch
from gridcast.intervals.quantile import interval_quantiles, quantile_key, quantile_levels
from gridcast.logs import get_logger
from gridcast.provenance import git_sha

log = get_logger(__name__)

PERIOD_MINUTES = 30  # AEMO's PDPASA resolution (m3_plan.md §2.6)
STEP_MINUTES = 5
PERIOD_STEPS = PERIOD_MINUTES // STEP_MINUTES
CONFORMAL = "conformal30"
CQR = "cqr30"
CQR_POINT = "lgbm_C"  # CQR corrects quantile LightGBM C; its point forecast is the WIS median


def conformal_name(model: str) -> str:
    return f"{CONFORMAL}_{model}"


CQR_NAME = f"{CQR}_{CQR_POINT}"


# --------------------------------------------------------------------------- periods


def period_means(values: np.ndarray, period_steps: int = PERIOD_STEPS) -> np.ndarray:
    """(n, H) 5-minute values → (n, H / period_steps) period means (NaN if any value is)."""
    n, h = values.shape
    if h % period_steps:
        raise ValueError(f"{h} leads are not a whole number of {period_steps}-step periods")
    return (
        np.asarray(values, dtype=np.float64)
        .reshape(n, h // period_steps, period_steps)
        .mean(axis=2)
    )


def period_mask(mask: np.ndarray, period_steps: int = PERIOD_STEPS) -> np.ndarray:
    """(n, H) pair mask → (n, H / period_steps): a period counts only if all its pairs do."""
    n, h = mask.shape
    if h % period_steps:
        raise ValueError(f"{h} leads are not a whole number of {period_steps}-step periods")
    return np.asarray(mask, dtype=bool).reshape(n, h // period_steps, period_steps).all(axis=2)


def period_bands(
    bands: Sequence[Sequence[int]], period_steps: int = PERIOD_STEPS
) -> list[tuple[int, int]]:
    """5-minute lead bands → bands of whole periods (each band edge must fall on a period's)."""
    out = []
    for lo, hi in bands:
        if (lo - 1) % period_steps or hi % period_steps:
            raise ValueError(f"band {lo}-{hi} does not consist of whole periods")
        out.append(((lo - 1) // period_steps + 1, hi // period_steps))
    return out


@dataclass(slots=True)
class PeriodInputs:
    """Everything the half-hour runs read, on run 2's development issue times."""

    origin_times: pd.DatetimeIndex
    actual: np.ndarray  # (n, K) period means of the actuals
    mask: np.ndarray  # (n, K) scored and calibration periods
    points: dict[str, np.ndarray]  # point model -> (n, K) period-mean forecasts
    quantiles: dict[int, tuple[np.ndarray, np.ndarray]]  # level -> period-mean (q_lo, q_hi)
    groups: dict[str, tuple[np.ndarray, int]]

    def only(self, model: str | None = None, level: int | None = None) -> PeriodInputs:
        """The same inputs with one point model or one level's quantiles (for a worker)."""
        return PeriodInputs(
            self.origin_times,
            self.actual,
            self.mask,
            {model: self.points[model]} if model is not None else {},
            {level: self.quantiles[level]} if level is not None else {},
            self.groups,
        )


def period_groups(
    cfg: AppConfig, region_id: str, origin_times: pd.DatetimeIndex, n_periods: int
) -> dict[str, tuple[np.ndarray, int]]:
    """Group labels per period: one group (``lead``), or the time-of-day block of its start."""
    setup = ForecastSetup.from_config(cfg)
    if (setup.issue_every_steps * STEP_MINUTES) % PERIOD_MINUTES:
        raise ValueError("issue times must fall on period boundaries")
    return {
        "lead": (np.zeros((len(origin_times), n_periods), dtype=np.int8), 1),
        "lead_tod": tod_labels(
            origin_times,
            n_periods,
            setup.issue_every_steps * STEP_MINUTES // PERIOD_MINUTES,
            cfg.region(region_id).local_tz,
            cfg.intervals.tod_block_starts,
            step_minutes=PERIOD_MINUTES,
        ),
    }


def build_inputs(
    cfg: AppConfig,
    region_id: str,
    actual: np.ndarray,
    mask: np.ndarray,
    origin_times: pd.DatetimeIndex,
    points: Mapping[str, np.ndarray],
    quantiles: Mapping[str, np.ndarray],
) -> PeriodInputs:
    """Period means of the 5-minute actuals, forecasts and quantiles (keyed by quantile key)."""
    setup = ForecastSetup.from_config(cfg)
    if actual.shape[1] != setup.horizon_steps:
        raise ValueError("the stored batch does not have forecast.horizon_steps leads")
    y = period_means(actual)
    out_q = {}
    for level in cfg.intervals.levels:
        lo_key, hi_key = interval_quantiles(level)
        out_q[level] = (period_means(quantiles[lo_key]), period_means(quantiles[hi_key]))
    return PeriodInputs(
        origin_times=origin_times,
        actual=y,
        mask=period_mask(mask),
        points={m: period_means(v) for m, v in points.items()},
        quantiles=out_q,
        groups=period_groups(cfg, region_id, origin_times, y.shape[1]),
    )


# --------------------------------------------------------------------------- the runs


@dataclass(frozen=True, slots=True)
class PeriodTask:
    """One worker's job: conformal around one point model, or CQR at one level."""

    cfg: AppConfig
    inputs: PeriodInputs
    method: str  # CONFORMAL or CQR
    model: str | None = None  # conformal: the point model
    level: int | None = None  # CQR: the level
    end: pd.Timestamp | None = None


def _latency_periods(cfg: AppConfig) -> int:
    """Availability latency in whole periods, rounded up (0 for M2's settings)."""
    steps = ForecastSetup.from_config(cfg).latency_steps
    return -(-steps // PERIOD_STEPS)


def period_run(task: PeriodTask) -> ConformalResult:
    """Every candidate stream and the per-retrain selection on half-hour periods."""
    cfg, inp = task.cfg, task.inputs
    if task.end is None:
        raise ValueError("a period task needs the end of its issue times")
    setup = ForecastSetup.from_config(cfg)
    issue_periods = setup.issue_every_steps * STEP_MINUTES // PERIOD_MINUTES
    latency = _latency_periods(cfg)
    if task.method == CONFORMAL:
        if task.model is None:
            raise ValueError("a conformal task needs its point model")
        levels = list(cfg.intervals.levels)
        problem = point_problem(
            inp.origin_times,
            issue_periods,
            latency,
            inp.actual,
            inp.points[task.model],
            inp.mask,
            inp.groups,
            levels,
            step_minutes=PERIOD_MINUTES,
        )
    elif task.method == CQR:
        if task.level is None:
            raise ValueError("a CQR task needs its level")
        levels = [task.level]
        problem = quantile_problem(
            inp.origin_times,
            issue_periods,
            latency,
            inp.actual,
            {task.level: inp.quantiles[task.level]},
            inp.mask,
            inp.groups,
            step_minutes=PERIOD_MINUTES,
        )
    else:
        raise ValueError(f"unknown method {task.method!r}")
    log.info("intervals.halfhour.start", method=task.method, model=task.model, level=task.level)
    result = run_conformal(
        problem,
        candidate_grid(cfg),
        levels,
        selection_retrains(cfg, task.end),
        period_bands(cfg.ml.lead_bands),
        cfg.intervals.selection_weeks,
    )
    log.info("intervals.halfhour.done", method=task.method, model=task.model, level=task.level)
    return result


@dataclass(slots=True)
class HalfHourRun:
    batch_id: uuid.UUID
    run_ids: dict[str, int]  # run name -> run id
    results: dict[str, ConformalResult]  # run name -> result
    metrics: dict[str, pd.DataFrame]  # run name -> metrics
    run_config: dict[str, Any]


def load_period_inputs(
    engine: Engine, cfg: AppConfig, region_id: str, end: pd.Timestamp
) -> tuple[PeriodInputs, dict[str, Any], Any]:
    """Run 2's development batch and the QR batch, checked as for CQR, as period means."""
    source_config = check_source_batch(engine, cfg.intervals.source_batch)
    qr = check_qr_batch(engine, cfg)
    keys = [quantile_key(t) for t in quantile_levels(cfg.intervals.levels)]
    directory = cfg.data_path(cfg.paths.backtest_dir)
    batch = read_stored_batch(directory, cfg.intervals.source_batch, TARGET).restrict(end=end)
    check_alignment(batch, qr.directory, keys)
    missing = set(cfg.intervals.point_models) - set(batch.forecasts)
    if missing:
        raise KeyError(
            f"batch {cfg.intervals.source_batch} has no forecasts from {sorted(missing)}"
        )
    times, quantiles = read_qr_quantiles(qr.directory, keys)
    if not times.equals(batch.origin_times):
        raise ValueError("the QR quantiles' issue times differ from run 2's")
    inputs = build_inputs(
        cfg,
        region_id,
        batch.actual,
        batch.common_mask(),
        batch.origin_times,
        {m: batch.forecasts[m] for m in cfg.intervals.point_models},
        quantiles,
    )
    return inputs, source_config, qr


def run_intervals_halfhour(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    workers: int = 4,
    sha: str | None = None,
) -> HalfHourRun:
    """Build, score and store the half-hour interval batch (development)."""
    t_start = time.perf_counter()
    end = nem_midnight_utc(str(cfg.backtest.holdout_start))
    inputs, source_config, qr = load_period_inputs(engine, cfg, region_id, end)
    levels = list(cfg.intervals.levels)
    models = list(cfg.intervals.point_models)
    tasks = [PeriodTask(cfg, inputs.only(model=m), CONFORMAL, model=m, end=end) for m in models]
    tasks += [PeriodTask(cfg, inputs.only(level=lv), CQR, level=lv, end=end) for lv in levels]
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        parts = list(pool.map(period_run, tasks))
    results = {conformal_name(m): parts[i] for i, m in enumerate(models)}
    results[CQR_NAME] = merge_levels(parts[len(models) :], levels)

    times = results[CQR_NAME].origin_times
    rows = np.asarray(inputs.origin_times >= times[0])
    actual, mask = inputs.actual[rows], inputs.mask[rows]
    metrics = {
        conformal_name(m): bounds_metrics(
            results[conformal_name(m)].lower,
            results[conformal_name(m)].upper,
            actual,
            mask,
            inputs.points[m][rows],
        )
        for m in models
    }
    metrics[CQR_NAME] = cqr_metrics(results[CQR_NAME], actual, mask, inputs.points[CQR_POINT][rows])

    batch_id = uuid.uuid4()
    out_dir = cfg.data_path(cfg.intervals.output_dir) / str(batch_id)
    save_results(
        out_dir, results, actual, times, name_of=lambda name: name, actual_dtype=np.float64
    )
    run_config: dict[str, Any] = {
        "evaluation_set": EVALUATION_SET,
        "resolution_minutes": PERIOD_MINUTES,
        "period_bands": [list(b) for b in period_bands(cfg.ml.lead_bands)],
        "source_batch": cfg.intervals.source_batch,
        "source_config_hash": source_config.get("config_hash"),
        "qr_batch": qr.batch_id,
        "qr_run_id": qr.run_id,
        "qr_config_hash": qr.config.get("qr_config_hash"),
        "interval_config_hash": interval_config_hash(cfg),
        "intervals": cfg.intervals.model_dump(mode="json"),
        "lead_bands": [list(b) for b in cfg.ml.lead_bands],
        "retrains": [str(r) for r in selection_retrains(cfg, end)],
        "calibration_first_issue": str(inputs.origin_times[0]),
        "first_issue": str(times[0]),
        "last_issue": str(times[-1]),
        "n_origins": len(times),
        "scored_pairs": int(mask.sum()),
        "total_pairs": int(mask.size),
    }
    run_config["seconds"] = round(time.perf_counter() - t_start, 1)
    sha = sha or git_sha()
    holdout = nem_midnight_utc(str(cfg.backtest.holdout_start))
    point_of = {conformal_name(m): (CONFORMAL, m) for m in models} | {CQR_NAME: (CQR, CQR_POINT)}
    run_ids: dict[str, int] = {}
    with engine.begin() as conn:
        for name, res in results.items():
            method, point = point_of[name]
            config = {**run_config, "method": method, "point_model": point, **result_config(res)}
            run_ids[name] = insert_run(
                conn,
                batch_id,
                name,
                region_id,
                (times[0], times[-1]),
                holdout,
                config,
                sha,
                out_dir.relative_to(cfg.data_root()),
                metrics[name],
            )
    return HalfHourRun(batch_id, run_ids, results, metrics, run_config)
