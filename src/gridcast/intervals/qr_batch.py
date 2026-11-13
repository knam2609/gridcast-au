"""Quantile LightGBM C as one stored development batch (M3 step 2, ``gridcast backtest-qr``).

The walk-forward (:mod:`gridcast.intervals.quantile`) runs over M2's development issue
times (``ml.first_issue`` → ``backtest.holdout_start``) with run 2's per-retrain settings
(``intervals.source_batch``). The quantile forecasts before ``intervals.first_issue`` are
the calibration warm-up of CQR (step 3); metrics are computed only from
``intervals.first_issue``, on the same pairs as the conformal batch (run 2's common mask).

Output, under ``<data root>/<intervals.output_dir>/<batch id>/``:

* ``qr_lgbm_C__q025.parquet`` … ``__q975.parquet``: wide quantile forecasts after
  rearrangement (sorted per pair), every development issue time;
* ``actual.parquet``: the actuals of the same issue times.

Each quantile selects its own setting at every (retrain, band) by pinball loss
(:func:`~gridcast.intervals.quantile.select_quantile`; user decision 2026-11-11).

Model files go to ``<data root>/<ml.model_dir>/<batch id>/qr_lgbm_C/``. One ``model_runs``
row (kind ``interval``, name ``qr_lgbm_C``) records every quantile's candidates and
selection (pinball loss and tree count), the crossing rates before sorting and the point
reproduction. ``forecast_metrics`` holds coverage, share below/above, width and interval
score per level, the WIS (median = the point forecast), and per quantile the pinball loss
and the share of actuals at or below it (the reliability diagram), plus the crossing share,
pooled (horizon 0) and per lead.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from gridcast.config import AppConfig
from gridcast.features.inputs import load_context, load_feature_builder
from gridcast.forecasting.ml_backtest import MAIN_VARIANTS, ml_config_hash
from gridcast.forecasting.origins import nem_midnight_utc, origin_positions, target_positions
from gridcast.forecasting.walkforward import retrain_schedule
from gridcast.intervals.batch import (
    EVALUATION_SET,
    TARGET,
    bounds_metrics,
    check_source_batch,
    insert_run,
    interval_config_hash,
    per_lead_rows,
    wide,
)
from gridcast.intervals.inputs import read_stored_batch, read_stored_model
from gridcast.intervals.quantile import (
    QrResult,
    ReproductionError,
    crossed_pairs,
    interval_quantiles,
    point_fits,
    quantile_grid,
    quantile_key,
    quantile_levels,
    run_qr_walk_forward,
)
from gridcast.intervals.scores import pinball, rearrange
from gridcast.logs import get_logger
from gridcast.provenance import git_sha

log = get_logger(__name__)

METHOD = "qr"
POINT_MODEL = "lgbm_C"
VARIANT = "C"
RUN_NAME = f"{METHOD}_{POINT_MODEL}"


SELECTION = "per_quantile_grid"


def qr_config_hash(cfg: AppConfig) -> str:
    """Hash of everything that changes the QR forecasts: M2's ML recipe, the source batch
    (whose rows the point reproduction pins), the quantile levels, the selection rule and its
    grid, and the training group."""
    payload = {
        "ml_config_hash": ml_config_hash(cfg),
        "source_batch": cfg.intervals.source_batch,
        "quantiles": list(quantile_levels(cfg.intervals.levels)),
        "selection": SELECTION,
        "grid": quantile_grid(cfg.ml),
        "variant": VARIANT,
        "group": list(MAIN_VARIANTS),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(slots=True)
class QrRun:
    batch_id: uuid.UUID
    result: QrResult
    times: pd.DatetimeIndex  # issue times of the forecasts
    actual: np.ndarray  # (n, H) float64
    sorted_quantiles: dict[str, np.ndarray]  # quantile key -> (n, H) float32, rearranged
    run_config: dict[str, Any]
    seconds: float
    source_records: list[dict[str, Any]]  # run 2's lgbm_C fit records

    @property
    def complete(self) -> bool:
        return bool(self.run_config["complete_window"])


def sorted_by_key(result: QrResult) -> dict[str, np.ndarray]:
    """Rearranged quantiles (sorted per pair), keyed by quantile; NaN pairs stay NaN."""
    stacked = rearrange(result.quantiles)
    return {quantile_key(t): stacked[k] for k, t in enumerate(result.taus)}


def qr_metrics(
    run: QrRun,
    levels: list[int],
    actual: np.ndarray,
    mask: np.ndarray,
    rows: np.ndarray,
    point: np.ndarray,
) -> pd.DataFrame:
    """Every stored metric of the QR batch on issue-time ``rows`` and pairs ``mask``.

    ``actual``, ``mask`` and ``point`` cover the selected rows only.
    """
    q = {k: v[rows] for k, v in run.sorted_quantiles.items()}
    if any(np.any(mask & ~np.isfinite(v)) for v in q.values()):
        raise ValueError("a scored pair has no quantile forecast")
    lower, upper = {}, {}
    for level in levels:
        lo, hi = interval_quantiles(level)
        lower[level], upper[level] = q[lo], q[hi]
    out = [bounds_metrics(lower, upper, actual, mask, point)]
    extra: list[tuple[str, int, float, int]] = []
    for tau in run.result.taus:
        key = quantile_key(tau)
        values = q[key].astype(np.float64)
        with np.errstate(invalid="ignore"):
            extra += per_lead_rows(f"pinball_{key}", pinball(values, actual, tau), mask)
            extra += per_lead_rows(f"share_le_{key}", (actual <= values).astype(float), mask)
    _finite, crossed = crossed_pairs(run.result.quantiles)
    extra += per_lead_rows("crossing", crossed[rows].astype(float), mask)
    out.append(pd.DataFrame(extra, columns=["metric", "horizon", "value", "n"]))
    return pd.concat(out, ignore_index=True)


def run_backtest_qr(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    until: pd.Timestamp | None = None,
    model_root: Path | None = None,
) -> QrRun:
    """Fit, reproduce and forecast; nothing is stored here (see :func:`store_qr`).

    ``until`` ends the run early (a smoke run); ``model_root`` (normally ``<data
    root>/models``) receives the batch's model files.
    """
    t_start = time.perf_counter()
    source = cfg.intervals.source_batch
    source_config = check_source_batch(engine, source)
    current = ml_config_hash(cfg)
    if source_config.get("config_hash") != current:
        raise ValueError(
            f"run 2's ML config hash {source_config.get('config_hash')} differs from the "
            f"current {current}: QR must reuse the same recipe"
        )
    records = list(source_config["model_params"]["retrains"])
    fits = point_fits(records)
    taus = quantile_levels(cfg.intervals.levels)
    grid = quantile_grid(cfg.ml)

    ctx = load_context(engine, cfg, region_id)
    builder = load_feature_builder(engine, cfg, region_id, ctx=ctx)
    setup = builder.setup
    first = nem_midnight_utc(str(cfg.ml.first_issue))
    holdout = nem_midnight_utc(str(cfg.backtest.holdout_start))
    end = min(holdout, until) if until is not None else holdout
    origins = origin_positions(ctx, setup, first)
    origins = origins[ctx.times[origins] < end]
    if origins.size == 0:
        raise LookupError("no development issue times to forecast")
    times = pd.DatetimeIndex(ctx.times[origins])
    schedule = retrain_schedule(cfg.ml.first_issue, end, cfg.ml.retrain_every_months)

    directory = cfg.data_path(cfg.paths.backtest_dir)
    ref_times, ref_values = read_stored_model(directory, source, POINT_MODEL)
    sel = np.asarray((ref_times >= times[0]) & (ref_times <= times[-1]))
    if int(sel.sum()) != len(times) or not bool((ref_times[sel] == times).all()):
        raise ReproductionError("run 2's stored issue times differ from the QR run's")
    reference = ref_values[sel].astype(np.float32)

    batch_id = uuid.uuid4()
    model_dir = model_root / str(batch_id) / RUN_NAME if model_root is not None else None
    result = run_qr_walk_forward(
        builder,
        origins,
        schedule,
        cfg.ml,
        fits,
        taus,
        grid,
        variant=VARIANT,
        group=MAIN_VARIANTS,
        reference=reference,
        model_dir=model_dir,
        data_root=cfg.data_root() if model_dir is not None else None,
    )
    actual = builder.ctx.y[target_positions(origins, setup)].astype(np.float64)
    finite, crossed = crossed_pairs(result.quantiles)
    run_config: dict[str, Any] = {
        "evaluation_set": EVALUATION_SET,
        "method": METHOD,
        "point_model": POINT_MODEL,
        "variant": VARIANT,
        "training_group": list(MAIN_VARIANTS),
        "source_batch": source,
        "source_config_hash": source_config.get("config_hash"),
        "ml_config_hash": current,
        "qr_config_hash": qr_config_hash(cfg),
        "interval_config_hash": interval_config_hash(cfg),
        "quantiles": list(taus),
        "levels": list(cfg.intervals.levels),
        "selection": SELECTION,
        "grid": grid,
        "lead_bands": [list(b) for b in cfg.ml.lead_bands],
        "retrains": result.folds,
        "forecast_first_issue": str(times[0]),
        "last_issue": str(times[-1]),
        "n_origins": len(times),
        "complete_window": until is None,
        "train_rows": result.train_rows,
        "shared_train_rows": result.usable_rows,
        "point_reproduction": {
            "passed": True,  # a mismatch raises ReproductionError before this point
            "reference_batch": source,
            "issue_times": len(times),
            "pairs": int(np.isfinite(reference).sum()),
            "fits": len(result.fits),
        },
        "crossing_share_all_pairs": float(crossed.sum() / max(int(finite.sum()), 1)),
        "fits": [asdict(r) for r in result.fits],
    }
    seconds = time.perf_counter() - t_start
    run_config["seconds"] = round(seconds, 1)
    return QrRun(
        batch_id, result, times, actual, sorted_by_key(result), run_config, seconds, records
    )


def score_development(run: QrRun, cfg: AppConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Metrics on the interval development period, on run 2's common mask (complete run)."""
    first = nem_midnight_utc(str(cfg.intervals.first_issue))
    holdout = nem_midnight_utc(str(cfg.backtest.holdout_start))
    source = read_stored_batch(
        cfg.data_path(cfg.paths.backtest_dir), cfg.intervals.source_batch, TARGET
    ).restrict(first, holdout)
    rows = np.asarray(run.times >= first)
    times = run.times[rows]
    if len(times) != len(source.origin_times) or not bool((times == source.origin_times).all()):
        raise ValueError("the QR issue times do not match run 2's development issue times")
    stored = source.actual.astype(np.float32)
    if not np.array_equal(run.actual[rows].astype(np.float32), stored, equal_nan=True):
        raise ValueError("the QR actuals differ from run 2's stored actuals")
    # Scored like the conformal batch: run 2's stored (float32) actuals and common mask.
    mask = source.common_mask()
    point = source.forecasts[POINT_MODEL]
    metrics = qr_metrics(run, list(cfg.intervals.levels), source.actual, mask, rows, point)
    summary = {
        "first_issue": str(times[0]),
        "scored_pairs": int(mask.sum()),
        "total_pairs": int(mask.size),
    }
    return metrics, summary


def smoke_metrics(run: QrRun, cfg: AppConfig) -> pd.DataFrame:
    """Pooled raw-QR metrics of a smoke run on every pair it can score (pre-scoring data)."""
    rows = np.ones(len(run.times), dtype=bool)
    mask = np.isfinite(run.actual) & np.isfinite(run.result.point)
    for v in run.sorted_quantiles.values():
        mask &= np.isfinite(v)
    point = run.result.point.astype(np.float64)
    metrics = qr_metrics(run, list(cfg.intervals.levels), run.actual, mask, rows, point)
    return metrics[metrics["horizon"] == 0]


def cost_estimate(run: QrRun) -> dict[str, float]:
    """Hours for the full development run, extrapolated from the smoke run's first retrain.

    Fit time per band is scaled two ways over the 12 retrains: by run 2's training rows
    (``rows``), and by run 2's own fit time growth for lgbm_C's chosen setting (``run2``,
    which also captures growing tree counts). Prediction time scales with the issue times.
    """
    first = run.result.folds[0]
    smoke_fit: dict[str, float] = {}
    for rec in run.result.fits:
        if rec.retrain == first:
            smoke_fit[rec.band] = rec.point_refit_seconds + sum(
                q.score_seconds + q.refit_seconds for q in rec.quantiles
            )
    rows: dict[str, dict[str, float]] = {}
    secs: dict[str, dict[str, float]] = {}
    for r in run.source_records:
        chosen = next(c for c in r["candidates"] if c["setting"] == r["chosen"])
        rows.setdefault(r["band"], {})[r["retrain"]] = float(r["rows"])
        secs.setdefault(r["band"], {})[r["retrain"]] = float(chosen["seconds"]) + float(
            r["refit_seconds"]
        )
    by_rows = sum(t * sum(rows[b].values()) / rows[b][first] for b, t in smoke_fit.items())
    by_run2 = sum(t * sum(secs[b].values()) / secs[b][first] for b, t in smoke_fit.items())
    n_retrains = len(next(iter(rows.values())))
    fit_smoke = sum(smoke_fit.values())
    other = max(run.seconds - fit_smoke, 0.0)  # loading, sample, prediction, checks
    overhead = other * n_retrains  # generous: the sample is built once, not per retrain
    return {
        "smoke_hours": run.seconds / 3600,
        "smoke_fit_hours": fit_smoke / 3600,
        "retrains": float(n_retrains),
        "full_hours_by_rows": (by_rows + overhead) / 3600,
        "full_hours_by_run2": (by_run2 + overhead) / 3600,
    }


def save_qr(directory: Path, run: QrRun) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    wide(run.times, run.actual.astype(np.float32)).to_parquet(
        directory / "actual.parquet", index=False, compression="zstd"
    )
    for key, values in run.sorted_quantiles.items():
        wide(run.times, values).to_parquet(
            directory / f"{RUN_NAME}__{key}.parquet", index=False, compression="zstd"
        )


def store_qr(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    run: QrRun,
    metrics: pd.DataFrame,
    summary: dict[str, Any],
    sha: str | None = None,
) -> int:
    """Write the parquet files and the ``model_runs``/``forecast_metrics`` rows."""
    if not run.complete:
        raise ValueError("a smoke run (--until) is never stored")
    out_dir = cfg.data_path(cfg.intervals.output_dir) / str(run.batch_id)
    save_qr(out_dir, run)
    holdout = nem_midnight_utc(str(cfg.backtest.holdout_start))
    first = pd.Timestamp(summary["first_issue"])
    config = {**run.run_config, **summary}
    with engine.begin() as conn:
        return insert_run(
            conn,
            run.batch_id,
            RUN_NAME,
            region_id,
            (first, run.times[-1]),
            holdout,
            config,
            sha or git_sha(),
            out_dir.relative_to(cfg.data_root()),
            metrics,
        )
