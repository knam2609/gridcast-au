"""Quantile LightGBM walk-forward for variant C (method 2, m3_plan.md §2.3).

The quantile models follow M2's point model as closely as possible:

* **Same harness pieces:** retrain schedule, staggered training sample, the rows shared by
  M2's group A, C and D (:func:`~gridcast.forecasting.walkforward.shared_usable`), the
  training-row and early-stopping rule at each cutoff, lead bands and fold forecasts all
  come from :mod:`gridcast.forecasting.walkforward`, unchanged. Only the loop is new: it
  fits several quantiles per (retrain, band) and trains variant C alone on the group's
  shared rows, which ``run_walk_forward`` cannot do.
* **Per-quantile selection** (user decision 2026-11-11, after the step-2 smoke run; it
  replaces copying run 2's point settings): at each (retrain, band) every quantile scores
  every candidate setting (target form × ``num_leaves``, M2's grid with the objective fixed
  to ``quantile``) by its **pinball loss** on the retrain's early-stopping slice (LightGBM's
  ``quantile`` metric is the mean pinball loss). The lowest wins, ties to the earliest in
  grid order, and only it is refit on every training row with its early-stopped tree
  count. A quantile of ``y − anchor`` plus an anchor known at the issue time is a quantile
  of ``y``, so every target form applies.

**Point reproduction.** At every (retrain, band) the loop also refits run 2's point setting
with its stored tree count. Its row counts and cutoff must match run 2's record before any
quantile is fitted, and its fold forecasts must equal run 2's stored forecasts exactly
(:class:`ReproductionError` otherwise). That proves the quantiles train on the point
model's rows, not merely on rows built the same way.

The new code lives under ``gridcast.intervals`` rather than next to ``models/ml.py`` so that
the frozen M2 paths (``evaluation.holdout.FROZEN_PATHS``) stay unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gridcast.config import LightGBMConfig, MlConfig
from gridcast.features.builder import FeatureBuilder
from gridcast.forecasting.walkforward import (
    BandedModel,
    band_key,
    fit_key,
    fit_rows,
    grid_position,
    predict_origins,
    shared_usable,
    train_origins,
    training_sample,
)
from gridcast.logs import get_logger
from gridcast.models.baselines import DAY_STEPS
from gridcast.models.ml import LightGBMModel, make_model
from gridcast.timeutils import NEM_TZ

log = get_logger(__name__)

POINT = "point"  # key of the reproduced point model in fold predictions


def quantile_levels(levels: Sequence[int]) -> tuple[float, ...]:
    """Quantile levels of central intervals at ``levels`` percent, ascending.

    50, 80, 95 → 0.025, 0.1, 0.25, 0.75, 0.9, 0.975.
    """
    taus = set()
    for level in levels:
        if not 0 < level < 100:
            raise ValueError(f"level must be in (0, 100), got {level}")
        half = level / 200.0
        taus |= {round(0.5 - half, 6), round(0.5 + half, 6)}
    return tuple(sorted(taus))


def quantile_key(tau: float) -> str:
    """``q025`` for 0.025, ``q900`` for 0.9 (per mille, three digits)."""
    permille = round(tau * 1000)
    if not 0 < permille < 1000 or abs(permille - tau * 1000) > 1e-6:
        raise ValueError(f"quantile {tau} is not a whole number of per mille in (0, 1)")
    return f"q{permille:03d}"


def interval_quantiles(level: int) -> tuple[str, str]:
    """Keys of the lower and upper quantile of the central interval at ``level``."""
    half = level / 200.0
    return quantile_key(round(0.5 - half, 6)), quantile_key(round(0.5 + half, 6))


def quantile_grid(cfg: MlConfig) -> list[dict[str, Any]]:
    """Settings each quantile searches, in a fixed order: M2's target forms × leaf counts."""
    return [
        {"target_form": form, "num_leaves": leaves}
        for form in cfg.grid.target_forms
        for leaves in cfg.grid.lightgbm_num_leaves
    ]


# --------------------------------------------------------------------------- the model


class QuantileLightGBMModel(LightGBMModel):
    """M2's LightGBM model with the ``quantile`` objective at level ``tau``.

    :meth:`score` returns the early-stopping **pinball loss** in the ``Score.mae`` field
    (the field name comes from the M2 interface); :class:`QuantileFit` stores it as
    ``es_pinball``.
    """

    family = "lightgbm_quantile"

    def __init__(
        self,
        columns: Sequence[str],
        target_form: str,
        tau: float,
        num_leaves: int,
        cfg: LightGBMConfig,
        seed: int,
    ) -> None:
        if not 0 < tau < 1:
            raise ValueError(f"quantile level must be in (0, 1), got {tau}")
        super().__init__(columns, target_form, "quantile", num_leaves, cfg, seed)
        self.tau = tau

    def _lgb_params(self) -> dict[str, Any]:
        params = super()._lgb_params()
        params.update({"objective": "quantile", "alpha": self.tau, "metric": "quantile"})
        return params

    def _score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        actual: np.ndarray,
        es_train: np.ndarray,
        es_valid: np.ndarray,
    ) -> tuple[float, int | None]:
        import lightgbm as lgb

        dtrain = lgb.Dataset(X[es_train], y[es_train], feature_name=self.columns)
        dvalid = lgb.Dataset(X[es_valid], y[es_valid], reference=dtrain)
        stopped = lgb.train(
            self._lgb_params(),
            dtrain,
            num_boost_round=self.cfg.max_rounds,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False)],
        )
        self.n_trees = max(int(stopped.best_iteration), 1)
        # The pinball loss of y − anchor equals that of y (the anchor shifts both sides).
        return float(stopped.best_score["valid_0"]["quantile"]), self.n_trees

    def params(self) -> dict[str, Any]:
        return {**super().params(), "family": self.family, "quantile": self.tau}


# --------------------------------------------------------------------------- records


@dataclass(frozen=True, slots=True)
class PointFit:
    """Run 2's stored fit of the point model at one (retrain, band)."""

    chosen: dict[str, Any]  # target_form, objective, num_leaves
    n_trees: int
    cutoff: str
    rows: int
    es_train_rows: int
    es_valid_rows: int

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> PointFit:
        return cls(
            chosen=dict(record["chosen"]),
            n_trees=int(record["params"]["n_trees"]),
            cutoff=str(record["cutoff"]),
            rows=int(record["rows"]),
            es_train_rows=int(record["es_train_rows"]),
            es_valid_rows=int(record["es_valid_rows"]),
        )


def point_fits(records: Sequence[Mapping[str, Any]]) -> dict[str, PointFit]:
    """``fit_key(retrain, band) -> PointFit`` from stored ``RetrainRecord`` dicts."""
    out = {fit_key(str(r["retrain"]), str(r["band"])): PointFit.from_record(r) for r in records}
    if len(out) != len(records):
        raise ValueError("duplicate (retrain, band) in the point model's records")
    return out


@dataclass(slots=True)
class QuantileFit:
    """One quantile at one (retrain, band): every candidate scored, the winner refit."""

    tau: float
    chosen: dict[str, Any]  # winning setting (target_form, num_leaves)
    n_trees: int  # the winner's early-stopped tree count
    es_pinball: float  # the winner's mean pinball loss (MW) on the early-stopping slice
    candidates: list[dict[str, Any]]  # setting, pinball, n_trees, seconds of every candidate
    score_seconds: float  # all candidates
    refit_seconds: float
    path: str | None  # model file, relative to the data root


@dataclass(slots=True)
class QrFitRecord:
    """One (retrain, band): the point reproduction and every quantile's selection."""

    retrain: str
    band: str
    cutoff: str
    fold_origins: int
    rows: int
    es_train_rows: int
    es_valid_rows: int
    point_chosen: dict[str, Any]  # run 2's point setting, refit for the reproduction check
    point_n_trees: int
    point_refit_seconds: float
    quantiles: list[QuantileFit]
    crossing_share: float = float("nan")  # fold pairs with crossed quantiles, before sorting


@dataclass(slots=True)
class QrResult:
    origins: np.ndarray  # evaluation origin positions
    taus: tuple[float, ...]
    quantiles: np.ndarray  # (n_tau, n_origins, H) float32, as fitted (not rearranged)
    point: np.ndarray  # (n_origins, H) float32: run 2's point forecasts, reproduced
    folds: list[str] = field(default_factory=list)
    fits: list[QrFitRecord] = field(default_factory=list)
    train_rows: int = 0  # rows of the training sample
    usable_rows: int = 0  # rows every variant of the group can use

    def by_key(self) -> dict[str, np.ndarray]:
        return {quantile_key(t): self.quantiles[k] for k, t in enumerate(self.taus)}


class ReproductionError(RuntimeError):
    """The loop did not reproduce run 2's point fit or forecasts; nothing is stored."""


def _check_rows(key: str, ref: PointFit, cutoff: str, rows: int, es_tr: int, es_va: int) -> None:
    got = {"cutoff": cutoff, "rows": rows, "es_train_rows": es_tr, "es_valid_rows": es_va}
    want = {
        "cutoff": ref.cutoff,
        "rows": ref.rows,
        "es_train_rows": ref.es_train_rows,
        "es_valid_rows": ref.es_valid_rows,
    }
    if got != want:
        diff = {k: (want[k], got[k]) for k in got if got[k] != want[k]}
        raise ReproductionError(f"{key}: training rows differ from run 2's (stored, now) {diff}")


def check_fold(label: str, reference: np.ndarray, point: np.ndarray) -> None:
    """Exact equality of the reproduced point forecasts with run 2's (NaN equal)."""
    if reference.shape != point.shape:
        raise ReproductionError(f"{label}: reference shape {reference.shape} != {point.shape}")
    same = (reference == point) | (np.isnan(reference) & np.isnan(point))
    if not same.all():
        gap = np.nanmax(np.abs(reference.astype(np.float64) - point.astype(np.float64)))
        raise ReproductionError(
            f"{label}: {int((~same).sum()):,} of {same.size:,} point forecasts differ from "
            f"run 2's (max |diff| {gap:.6g} MW)"
        )


def crossed_pairs(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(finite, crossed)`` for quantiles stacked on axis 0 (levels ascending): pairs with
    every quantile finite, and finite pairs where some quantile lies below a lower one."""
    finite = np.isfinite(q).all(axis=0)
    with np.errstate(invalid="ignore"):
        crossed = (np.diff(q, axis=0) < 0).any(axis=0) & finite
    return finite, crossed


def select_quantile(
    columns: Sequence[str],
    X: np.ndarray,
    actual: np.ndarray,
    es_train: np.ndarray,
    es_valid: np.ndarray,
    tau: float,
    candidates: Sequence[Mapping[str, Any]],
    cfg: MlConfig,
) -> tuple[QuantileLightGBMModel, QuantileFit]:
    """Score every candidate at ``tau`` by pinball loss; refit the lowest (first on ties).

    The returned fit's ``path`` is filled in by the caller.
    """
    if not candidates:
        raise ValueError("need at least one candidate setting")

    def model(choice: Mapping[str, Any]) -> QuantileLightGBMModel:
        return QuantileLightGBMModel(
            columns,
            str(choice["target_form"]),
            float(tau),
            int(choice["num_leaves"]),
            cfg.lightgbm,
            cfg.seed,
        )

    # Each candidate is fitted and dropped before the next, to bound memory.
    scores = [model(c).score(X, actual, es_train, es_valid) for c in candidates]
    best = min(range(len(scores)), key=lambda i: scores[i].mae)  # first on ties
    final = model(candidates[best])
    refit_seconds = final.refit(X, actual, scores[best].best_iteration)
    assert final.n_trees is not None
    fit = QuantileFit(
        tau=float(tau),
        chosen=dict(candidates[best]),
        n_trees=final.n_trees,
        es_pinball=scores[best].mae,
        candidates=[
            {
                "setting": dict(c),
                "pinball": sc.mae,
                "n_trees": sc.best_iteration,
                "seconds": sc.seconds,
            }
            for c, sc in zip(candidates, scores, strict=True)
        ],
        score_seconds=round(sum(sc.seconds for sc in scores), 2),
        refit_seconds=refit_seconds,
        path=None,
    )
    return final, fit


# --------------------------------------------------------------------------- walk-forward


def run_qr_walk_forward(
    builder: FeatureBuilder,
    eval_origins: np.ndarray,
    retrains: Sequence[pd.Timestamp],
    cfg: MlConfig,
    fits: Mapping[str, PointFit],
    taus: Sequence[float],
    candidates: Sequence[Mapping[str, Any]],
    variant: str = "C",
    group: Sequence[str] = ("A", "C", "D"),
    reference: np.ndarray | None = None,
    model_dir: Path | None = None,
    data_root: Path | None = None,
) -> QrResult:
    """Select and fit every quantile at each retrain and band; forecast each fold.

    ``candidates`` are the settings each quantile searches (:func:`quantile_grid`). ``fits``
    holds run 2's point fit per ``fit_key``; ``reference`` (``(n_origins, H)``, the stored
    point forecasts of ``eval_origins``) turns on the exact forecast check per fold.
    """
    if not retrains or not taus:
        raise ValueError("need at least one retrain and one quantile")
    if variant not in group:
        raise ValueError("the trained variant must belong to the group")
    if list(taus) != sorted(set(taus)):
        raise ValueError("quantile levels must be ascending and distinct")
    ctx, setup = builder.ctx, builder.setup
    eval_times = pd.DatetimeIndex(ctx.times[eval_origins])
    if eval_times.size == 0 or eval_times[0] < retrains[0]:
        raise ValueError("every evaluation issue time must follow the first retrain")
    if reference is not None and reference.shape != (len(eval_origins), setup.horizon_steps):
        raise ValueError("reference must have one row per evaluation issue time")
    cutoffs = [grid_position(builder, r) - setup.latency_steps for r in retrains]
    es_steps = cfg.early_stopping_weeks * 7 * DAY_STEPS

    origins = train_origins(builder, cutoffs[-1])
    samples = {v: training_sample(builder, v, origins, cfg.sample_every_steps) for v in group}
    usable = shared_usable(list(samples.values()))
    sample = samples[variant]
    del samples  # only the trained variant's rows are kept
    log.info(
        "qr.sample",
        variant=variant,
        group=list(group),
        rows=len(usable),
        usable=int(usable.sum()),
        first_issue=str(ctx.times[origins[0]]),
    )
    H = setup.horizon_steps
    result = QrResult(
        origins=eval_origins,
        taus=tuple(taus),
        quantiles=np.full((len(taus), len(eval_origins), H), np.nan, np.float32),
        point=np.full((len(eval_origins), H), np.nan, np.float32),
        train_rows=len(sample.X),
        usable_rows=int(usable.sum()),
    )
    keys = [quantile_key(t) for t in taus]
    for i, (retrain, cutoff) in enumerate(zip(retrains, cutoffs, strict=True)):
        in_fold = eval_times >= retrain
        if i + 1 < len(retrains):
            in_fold &= eval_times < retrains[i + 1]
        if not in_fold.any():
            continue
        label = retrain.tz_convert(NEM_TZ).strftime("%Y-%m-%d")
        result.folds.append(label)
        fold = eval_origins[in_fold]
        rows, es_train, es_valid = fit_rows(sample, usable, cutoff, es_steps)
        horizons = sample.horizons[rows]
        cutoff_label = str(ctx.times[cutoff])
        models: dict[str, BandedModel] = {k: BandedModel() for k in (POINT, *keys)}
        records: list[QrFitRecord] = []
        for band in cfg.lead_bands:
            bkey = band_key(band)
            key = fit_key(label, bkey)
            if key not in fits:
                raise KeyError(f"run 2 stored no point fit for {key}")
            ref = fits[key]
            in_band = (horizons >= band[0]) & (horizons <= band[1])
            X, actual = sample.X[rows[in_band]], sample.actual[rows[in_band]]
            tr, va = es_train[in_band], es_valid[in_band]
            _check_rows(key, ref, cutoff_label, len(X), int(tr.sum()), int(va.sum()))

            point = make_model("lightgbm", sample.columns, ref.chosen, cfg)
            point_seconds = point.refit(X, actual, ref.n_trees)
            models[POINT].bands[band] = point

            q_fits: list[QuantileFit] = []
            for tau, qkey in zip(taus, keys, strict=True):
                model, q_fit = select_quantile(
                    sample.columns, X, actual, tr, va, tau, candidates, cfg
                )
                if model_dir is not None:
                    written = model.save(model_dir / f"{label}_{bkey}_{qkey}")
                    q_fit.path = str(written.relative_to(data_root) if data_root else written)
                q_fits.append(q_fit)
                models[qkey].bands[band] = model
                log.info(
                    "qr.fit",
                    retrain=label,
                    band=bkey,
                    quantile=qkey,
                    chosen=q_fit.chosen,
                    n_trees=q_fit.n_trees,
                    es_pinball=round(q_fit.es_pinball, 3),
                    seconds=round(q_fit.score_seconds + q_fit.refit_seconds, 1),
                )
            records.append(
                QrFitRecord(
                    retrain=label,
                    band=bkey,
                    cutoff=cutoff_label,
                    fold_origins=len(fold),
                    rows=len(X),
                    es_train_rows=int(tr.sum()),
                    es_valid_rows=int(va.sum()),
                    point_chosen=dict(ref.chosen),
                    point_n_trees=ref.n_trees,
                    point_refit_seconds=point_seconds,
                    quantiles=q_fits,
                )
            )
            del X, actual
        t0 = time.perf_counter()
        preds = predict_origins(builder, variant, fold, models)
        result.point[in_fold] = preds[POINT]
        if reference is not None:
            check_fold(label, reference[in_fold], preds[POINT])
        stacked = np.stack([preds[k] for k in keys])
        result.quantiles[:, in_fold] = stacked
        finite, crossed = crossed_pairs(stacked)
        fold_h = np.arange(1, H + 1)
        for rec, band in zip(records, cfg.lead_bands, strict=True):
            cols = (fold_h >= band[0]) & (fold_h <= band[1])
            n = int(finite[:, cols].sum())
            rec.crossing_share = float(crossed[:, cols].sum() / n) if n else float("nan")
        result.fits.extend(records)
        log.info(
            "qr.fold",
            retrain=label,
            origins=len(fold),
            reproduced=reference is not None,
            crossing_share=round(float(crossed.sum() / max(int(finite.sum()), 1)), 4),
            predict_seconds=round(time.perf_counter() - t0, 1),
        )
    return result
