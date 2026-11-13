"""Quantile LightGBM C over the holdout, unscored (M3 step 6a, ``gridcast backtest-qr-holdout``).

The first of step 6's two stages (m3_plan.md step 6). It continues step 2's QR walk-forward
from the last development retrain (2025-08-01) through the holdout, with the frozen recipe
(step 2's per-quantile selection), and stores the forecasts. **Nothing is scored here:** no
metric is computed or logged. ``gridcast evaluate-holdout-intervals`` (step 6b, the U5 lock)
scores them, together with every other holdout interval.

* **Refusals first:** the interval freeze must hold (both recipe hashes, frozen code
  unchanged since ``intervals.frozen_commit``), the working tree must be clean, step 2's
  development QR batch and M2's ``holdout_main`` batch (of the frozen M2 recipe) must be
  stored, and no QR holdout batch may exist for the QR recipe. The data must cover every
  issue time with its full horizon (M2's rule).
* **The first retrain alone, then its reproduction check.** The 2025-08-01 retrain is
  fitted on its own first. Its fit records (rows, every candidate's pinball loss and tree
  count, the selection) and its rearranged quantiles for the development issue times
  (Aug–Sep 2025) must equal step 2's stored ones exactly; otherwise
  :class:`QrHoldoutReproductionFailed` is raised before any other retrain is fitted, and
  nothing is stored.
* **Point refits:** at every retrain the loop refits ``holdout_main``'s point setting with
  its stored tree count, checks the training rows against its records, and checks the fold
  forecasts against run 2's (development issue times) and ``holdout_main``'s (holdout issue
  times) exactly (:func:`~gridcast.intervals.quantile.run_qr_walk_forward`).

Output, under ``<data root>/<intervals.output_dir>/<batch id>/``, laid out as step 2's, for
issue times from the first retrain to ``backtest.holdout_end``:
``qr_lgbm_C__q025.parquet`` … ``__q975.parquet`` (rearranged), ``qr_lgbm_C__crossed.parquet``
(per pair: did the quantiles cross before sorting?) and ``actual.parquet``. Model files go to
``<data root>/<ml.model_dir>/<batch id>/qr_lgbm_C/``. One ``model_runs`` row (kind
``interval``, evaluation set ``intervals_holdout_qr``) records the fits and the reproduction,
with no ``forecast_metrics`` rows.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from gridcast.config import AppConfig
from gridcast.evaluation.holdout import (
    HOLDOUT_SETS,
    HoldoutRefused,
    HoldoutWindow,
    _holdout_origins,
    holdout_window,
)
from gridcast.features.inputs import load_context, load_feature_builder
from gridcast.forecasting.ml_backtest import MAIN_VARIANTS, ml_config_hash
from gridcast.forecasting.origins import target_positions
from gridcast.intervals.batch import TARGET, check_source_batch, interval_config_hash, wide
from gridcast.intervals.cqr_batch import QrSource, check_qr_batch
from gridcast.intervals.freeze import FreezeState
from gridcast.intervals.inputs import read_stored_model, read_wide
from gridcast.intervals.qr_batch import (
    POINT_MODEL,
    RUN_NAME,
    SELECTION,
    VARIANT,
    qr_config_hash,
    sorted_by_key,
)
from gridcast.intervals.quantile import (
    QrFitRecord,
    QrResult,
    crossed_pairs,
    point_fits,
    quantile_grid,
    quantile_key,
    quantile_levels,
    run_qr_walk_forward,
)
from gridcast.logs import get_logger

log = get_logger(__name__)

EVALUATION_SET = "intervals_holdout_qr"
CROSSED = "crossed"  # file suffix of the per-pair crossing flags
# Fields of a stored QR fit record that must reproduce at the first retrain. Timings, model
# paths, the fold size and the crossing share (the holdout run's first fold also holds
# holdout issue times) are left out.
RECORD_FIELDS = (
    "retrain",
    "band",
    "cutoff",
    "rows",
    "es_train_rows",
    "es_valid_rows",
    "point_chosen",
    "point_n_trees",
)
QUANTILE_FIELDS = ("tau", "chosen", "n_trees", "es_pinball")
CANDIDATE_FIELDS = ("setting", "pinball", "n_trees")


class QrHoldoutRefused(RuntimeError):
    """A precondition of the QR holdout run is not met; nothing was fitted."""


class QrHoldoutReproductionFailed(RuntimeError):
    """The first retrain does not reproduce step 2's QR fits or quantiles; nothing stored."""


# --------------------------------------------------------------------------- preconditions


@dataclass(frozen=True, slots=True)
class PointBatch:
    """M2's stored ``holdout_main`` batch: its id and lgbm_C's fit records."""

    batch_id: str
    records: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class QrHoldoutInputs:
    window: HoldoutWindow
    dev_qr: QrSource
    holdout_point: PointBatch
    freeze: FreezeState


def holdout_point_batch(engine: Engine, region_id: str, config_hash: str) -> PointBatch:
    """The latest ``holdout_main`` lgbm_C run of the frozen M2 recipe."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT batch_id::text AS batch, config FROM model_runs
                WHERE region_id = :r AND target = :t AND model_name = :m
                  AND status = 'success'
                  AND config->>'evaluation_set' = :es AND config->>'config_hash' = :h
                ORDER BY run_id DESC LIMIT 1
                """
            ),
            {
                "r": region_id,
                "t": TARGET,
                "m": POINT_MODEL,
                "es": HOLDOUT_SETS["main"],
                "h": config_hash,
            },
        ).one_or_none()
    if row is None:
        raise LookupError(f"no stored holdout_main {POINT_MODEL} run for ML config {config_hash}")
    return PointBatch(str(row.batch), list(row.config["model_params"]["retrains"]))


def stored_qr_holdout(engine: Engine, region_id: str, qr_hash: str) -> list[str]:
    """QR holdout batches already stored for the QR recipe ``qr_hash``."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT batch_id::text FROM model_runs
                WHERE region_id = :r AND model_name = :m AND config->>'evaluation_set' = :es
                  AND config->>'qr_config_hash' = :h
                """
            ),
            {"r": region_id, "m": RUN_NAME, "es": EVALUATION_SET, "h": qr_hash},
        )
        return sorted(str(b) for b in rows.scalars())


def preflight(
    engine: Engine, cfg: AppConfig, region_id: str, freeze: FreezeState
) -> QrHoldoutInputs:
    """Every refusal check; returns what the run reads."""
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
    dev_qr: QrSource | None = None
    try:
        dev_qr = check_qr_batch(engine, cfg)
    except (LookupError, ValueError) as exc:
        problems.append(f"development QR batch: {exc}")
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
    if stored := stored_qr_holdout(engine, region_id, freeze.qr_config_hash):
        problems.append(
            f"a QR holdout batch already exists for QR recipe {freeze.qr_config_hash}: "
            f"{', '.join(stored)}"
        )
    if problems:
        raise QrHoldoutRefused("; ".join(problems))
    assert window is not None and dev_qr is not None and point is not None
    if len(window.retrains) < 2:
        raise QrHoldoutRefused("the holdout needs a retrain after the last development one")
    return QrHoldoutInputs(window, dev_qr, point, freeze)


# --------------------------------------------------------------------------- reproduction


@dataclass(slots=True)
class QrReproduction:
    retrain: str
    reference_batch: str  # step 2's QR batch
    issue_times: int  # development issue times compared
    pairs: int  # finite quantile values compared, over every quantile
    fits: int  # (band) records compared
    mismatches: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"passed": self.passed}


def _fit_view(record: Mapping[str, Any]) -> dict[str, Any]:
    """The reproducible part of a QR fit record (stored JSON or :class:`QrFitRecord`)."""
    return {
        **{k: record[k] for k in RECORD_FIELDS},
        "quantiles": [
            {
                **{k: q[k] for k in QUANTILE_FIELDS},
                "candidates": [{k: c[k] for k in CANDIDATE_FIELDS} for c in q["candidates"]],
            }
            for q in record["quantiles"]
        ],
    }


def _normalise(view: Mapping[str, Any]) -> Any:
    return json.loads(json.dumps(view, sort_keys=True, default=str))


def check_first_retrain(
    result: QrResult,
    times: pd.DatetimeIndex,
    holdout_start: pd.Timestamp,
    dev_qr: QrSource,
) -> QrReproduction:
    """The first retrain's fits and development-period quantiles against step 2's batch."""
    label = result.folds[0]
    dev_rows = np.asarray(times < holdout_start)
    rep = QrReproduction(label, dev_qr.batch_id, int(dev_rows.sum()), 0, 0)

    stored = [f for f in dev_qr.config.get("fits", []) if f.get("retrain") == label]
    mine = [asdict(f) for f in result.fits if f.retrain == label]
    rep.fits = len(mine)
    if len(stored) != len(mine):
        rep.mismatches.append(f"{len(mine)} band records at {label}; step 2 stored {len(stored)}")
    for want, got in zip(stored, mine, strict=False):
        if _normalise(_fit_view(want)) != _normalise(_fit_view(got)):
            rep.mismatches.append(f"fit record {label} {got['band']} differs")

    if rep.issue_times == 0:
        rep.mismatches.append("no development issue time to compare")
        return rep
    sorted_q = sorted_by_key(result)
    dev_times = times[dev_rows]
    for key, values in sorted_q.items():
        ref_times, ref = read_wide(dev_qr.directory / f"{RUN_NAME}__{key}.parquet")
        sel = np.asarray((ref_times >= dev_times[0]) & (ref_times <= dev_times[-1]))
        if int(sel.sum()) != len(dev_times) or not bool((ref_times[sel] == dev_times).all()):
            rep.mismatches.append(f"{key}: step 2's issue times differ")
            continue
        new = values[dev_rows].astype(np.float64)
        if not np.array_equal(new, ref[sel], equal_nan=True):
            same = (new == ref[sel]) | (np.isnan(new) & np.isnan(ref[sel]))
            rep.mismatches.append(f"{key}: {int((~same).sum()):,} quantile values differ")
        rep.pairs += int(np.isfinite(new).sum())
    return rep


# --------------------------------------------------------------------------- the run


@dataclass(slots=True)
class QrHoldoutRun:
    batch_id: uuid.UUID
    result: QrResult
    times: pd.DatetimeIndex
    actual: np.ndarray  # (n, H) float64, from the market data
    sorted_quantiles: dict[str, np.ndarray]  # rearranged, float32
    crossed: np.ndarray  # (n, H) bool: quantiles crossed before sorting
    reproduction: QrReproduction
    run_config: dict[str, Any]


def point_reference(cfg: AppConfig, inputs: QrHoldoutInputs, times: pd.DatetimeIndex) -> np.ndarray:
    """lgbm_C's stored forecasts of ``times``: run 2's before the holdout, then holdout_main's."""
    directory = cfg.data_path(cfg.paths.backtest_dir)
    parts = []
    start = inputs.window.holdout_start
    for batch, keep in (
        (cfg.intervals.source_batch, lambda t: (t >= times[0]) & (t < start)),
        (inputs.holdout_point.batch_id, lambda t: (t >= start) & (t <= times[-1])),
    ):
        t, values = read_stored_model(directory, batch, POINT_MODEL)
        sel = np.asarray(keep(t))
        parts.append((t[sel], values[sel]))
    ref_times = parts[0][0].append(parts[1][0])
    if len(ref_times) != len(times) or not bool((ref_times == times).all()):
        raise QrHoldoutRefused(
            "run 2's and holdout_main's stored issue times do not match the QR holdout run's"
        )
    return np.concatenate([parts[0][1], parts[1][1]]).astype(np.float32)


def _merge(first: QrResult, rest: QrResult) -> QrResult:
    return QrResult(
        origins=np.concatenate([first.origins, rest.origins]),
        taus=first.taus,
        quantiles=np.concatenate([first.quantiles, rest.quantiles], axis=1),
        point=np.concatenate([first.point, rest.point]),
        folds=[*first.folds, *rest.folds],
        fits=[*first.fits, *rest.fits],
        train_rows=rest.train_rows,  # the later sample, up to the last cutoff
        usable_rows=rest.usable_rows,
    )


def run_qr_holdout(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    inputs: QrHoldoutInputs,
    model_root: Path | None = None,
) -> QrHoldoutRun:
    """Fit the first retrain, check it, then the rest; nothing is scored or stored here."""
    t_start = time.perf_counter()
    window = inputs.window
    ctx = load_context(engine, cfg, region_id)
    builder = load_feature_builder(engine, cfg, region_id, ctx=ctx)
    origins = _holdout_origins(builder, window)  # refuses if the data end too early
    times = pd.DatetimeIndex(ctx.times[origins])
    reference = point_reference(cfg, inputs, times)
    fits = point_fits(inputs.holdout_point.records)
    taus = quantile_levels(cfg.intervals.levels)
    grid = quantile_grid(cfg.ml)
    retrains = list(window.retrains)
    batch_id = uuid.uuid4()
    model_dir = model_root / str(batch_id) / RUN_NAME if model_root is not None else None
    data_root = cfg.data_root() if model_dir is not None else None
    first_fold = np.asarray(times < retrains[1])
    log.info(
        "qr_holdout.start",
        first_retrain=window.first_label,
        retrains=len(retrains),
        issue_times=len(times),
        first_fold_issue_times=int(first_fold.sum()),
    )

    def walk(rows: np.ndarray, schedule: Sequence[pd.Timestamp]) -> QrResult:
        return run_qr_walk_forward(
            builder,
            origins[rows],
            schedule,
            cfg.ml,
            fits,
            taus,
            grid,
            variant=VARIANT,
            group=MAIN_VARIANTS,
            reference=reference[rows],
            model_dir=model_dir,
            data_root=data_root,
        )

    first = walk(first_fold, retrains[:1])
    rep = check_first_retrain(first, times[first_fold], window.holdout_start, inputs.dev_qr)
    if not rep.passed:
        raise QrHoldoutReproductionFailed(
            f"the {rep.retrain} retrain does not reproduce step 2's QR batch "
            f"({rep.reference_batch}); nothing fitted further or stored: "
            + "; ".join(rep.mismatches)
        )
    log.info(
        "qr_holdout.reproduced",
        retrain=rep.retrain,
        issue_times=rep.issue_times,
        quantile_values=rep.pairs,
        fits=rep.fits,
    )
    rest = walk(~first_fold, retrains[1:])
    result = _merge(first, rest)

    actual = ctx.y[target_positions(origins, builder.setup)].astype(np.float64)
    finite, crossed = crossed_pairs(result.quantiles)
    holdout_rows = int((times >= window.holdout_start).sum())
    run_config: dict[str, Any] = {
        "evaluation_set": EVALUATION_SET,
        "period": "holdout",
        "method": "qr",
        "point_model": POINT_MODEL,
        "variant": VARIANT,
        "training_group": list(MAIN_VARIANTS),
        "source_batch": cfg.intervals.source_batch,
        "holdout_point_batch": inputs.holdout_point.batch_id,
        "development_qr_batch": inputs.dev_qr.batch_id,
        "ml_config_hash": ml_config_hash(cfg),
        "qr_config_hash": qr_config_hash(cfg),
        "interval_config_hash": interval_config_hash(cfg),
        "frozen_commit": inputs.freeze.frozen_commit,
        "code_sha": inputs.freeze.sha,
        "quantiles": list(taus),
        "levels": list(cfg.intervals.levels),
        "selection": SELECTION,
        "grid": grid,
        "lead_bands": [list(b) for b in cfg.ml.lead_bands],
        "retrains": result.folds,
        "forecast_first_issue": str(times[0]),
        "holdout_start": str(window.holdout_start),
        "holdout_end": str(window.holdout_end),
        "last_issue": str(times[-1]),
        "n_origins": len(times),
        "holdout_issue_times": holdout_rows,
        "complete_window": True,
        "scored": False,  # step 6b scores these forecasts under the U5 lock
        "train_rows": result.train_rows,
        "shared_train_rows": result.usable_rows,
        "point_reproduction": {
            "passed": True,  # a mismatch raises ReproductionError before this point
            "reference_batches": [cfg.intervals.source_batch, inputs.holdout_point.batch_id],
            "issue_times": len(times),
            "pairs": int(np.isfinite(reference).sum()),
            "fits": len(result.fits),
        },
        "qr_reproduction": rep.as_dict(),
        "crossing_share_all_pairs": float(crossed.sum() / max(int(finite.sum()), 1)),
        "fits": [asdict(r) for r in result.fits],
    }
    run_config["seconds"] = round(time.perf_counter() - t_start, 1)
    return QrHoldoutRun(
        batch_id, result, times, actual, sorted_by_key(result), crossed, rep, run_config
    )


def fit_rows(records: Sequence[QrFitRecord]) -> list[list[Any]]:
    """One printable row per (retrain, band): each quantile's choice, trees and ES loss."""
    short = {"residual": "res", "direct": "dir", "anchored": "anc", "latest": "lat"}
    rows = []
    for rec in records:
        fit_seconds = sum(q.score_seconds + q.refit_seconds for q in rec.quantiles)
        rows.append(
            [
                rec.retrain,
                rec.band,
                rec.rows,
                *(
                    f"{short[q.chosen['target_form']]}/{q.chosen['num_leaves']} "
                    f"{q.n_trees} {q.es_pinball:.1f}"
                    for q in rec.quantiles
                ),
                100 * rec.crossing_share,
                rec.point_refit_seconds + fit_seconds,
            ]
        )
    return rows


def fit_headers(taus: Sequence[float]) -> list[str]:
    return ["retrain", "band", "rows", *(quantile_key(t) for t in taus), "crossed %", "fit s"]


# --------------------------------------------------------------------------- storage


def store_qr_holdout(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    run: QrHoldoutRun,
    sha: str | None,
) -> int:
    """Write the forecasts and one ``model_runs`` row, with no metrics; re-checks the lock."""
    qr_hash = str(run.run_config["qr_config_hash"])
    if stored := stored_qr_holdout(engine, region_id, qr_hash):
        raise QrHoldoutRefused(f"a QR holdout batch appeared during the run: {', '.join(stored)}")
    out_dir = cfg.data_path(cfg.intervals.output_dir) / str(run.batch_id)
    out_dir.mkdir(parents=True, exist_ok=False)
    wide(run.times, run.actual.astype(np.float32)).to_parquet(
        out_dir / "actual.parquet", index=False, compression="zstd"
    )
    for key, values in run.sorted_quantiles.items():
        wide(run.times, values).to_parquet(
            out_dir / f"{RUN_NAME}__{key}.parquet", index=False, compression="zstd"
        )
    wide(run.times, run.crossed).to_parquet(
        out_dir / f"{RUN_NAME}__{CROSSED}.parquet", index=False, compression="zstd"
    )
    with engine.begin() as conn:
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
                "batch": run.batch_id,
                "name": RUN_NAME,
                "target": TARGET,
                "region": region_id,
                "eval_start": run.times[0].to_pydatetime(),
                "eval_end": run.times[-1].to_pydatetime(),
                "holdout": pd.Timestamp(run.run_config["holdout_start"]).to_pydatetime(),
                "config": json.dumps(run.run_config, default=str),
                "sha": sha,
                "artifact": str(out_dir.relative_to(cfg.data_root())),
            },
        ).scalar_one()
    return int(run_id)
