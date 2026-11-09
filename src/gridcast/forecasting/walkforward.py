"""Walk-forward evaluation of the M2 ML models (m2_plan.md §2.4).

* **Retrains** happen at NEM midnight on ``first_issue`` and every ``retrain_every_months``
  after it. The model fitted at retrain *R* forecasts every issue time in [R, next R).
* **Training rows** come from a staggered sample: every issue time, with targets every
  ``sample_every_steps`` steps whose first horizon rotates across issue times (and shifts
  by one each day), so every lead time and every 5-minute slot is trained on.
* **Leakage rule:** at *R* a pair is a training row only if its target is known, i.e.
  ``t <= R − latency``. Early stopping monitors the pairs issued in the last
  ``early_stopping_weeks`` before the cutoff and fits on pairs whose *targets* precede
  that slice; the model is then refit on every training row with the early-stopped tree
  count. Nothing from the forecast fold is ever read.
* **Lead bands:** each model is one fit per lead band (``ml.lead_bands``), trained on the
  band's rows only and selected separately; forecasts take each horizon from its band.
  One global model let long-lead corrections swamp the short leads.
* **Selection at every retrain:** each candidate setting (target form, LightGBM objective
  and leaves, ridge alpha) is scored by its MAE on that retrain's early-stopping slice;
  the lowest wins (ties go to the earliest in grid order) and only it is refit. A model
  can instead follow fixed per-retrain settings (the publication-delay sensitivity run
  reuses headline C's choices, so only the input timing differs).
* **Shared rows:** variants evaluated together (A, C, D) train on the same rows: a pair
  is used only if *every* variant of the group has all its features, so a C-vs-D gap
  reflects the weather source alone.
* **Scoring rows:** forecasts are produced for every horizon 1…H of every issue time in
  the fold. Pairs with a missing feature get NaN and drop out of the common mask.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gridcast.config import MlConfig
from gridcast.features.builder import FeatureBuilder, FeatureMatrix
from gridcast.forecasting.backtest import monthly_folds
from gridcast.forecasting.origins import ForecastSetup, nem_midnight_utc
from gridcast.logs import get_logger
from gridcast.models.baselines import DAY_STEPS, WEEK_STEPS
from gridcast.models.ml import FAMILY_PREFIX, MlModel, Score, make_model
from gridcast.timeutils import INTERVAL, NEM_TZ

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model to walk forward: a variant, a family and how its settings are chosen.

    Exactly one of ``candidates`` (searched at every retrain and band) and ``fixed``
    (:func:`fit_key` of retrain and band -> setting) is given.
    """

    name: str
    variant: str
    family: str
    candidates: tuple[Mapping[str, Any], ...] = ()
    fixed: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        if bool(self.candidates) == (self.fixed is not None):
            raise ValueError(f"{self.name}: give either candidates or fixed settings")


def model_name(family: str, variant: str, suffix: str = "") -> str:
    return f"{FAMILY_PREFIX[family]}_{variant}{suffix}"


def band_key(band: tuple[int, int]) -> str:
    return f"h{band[0]:03d}-{band[1]:03d}"


def fit_key(retrain: str, band: str) -> str:
    """Key of one fit: retrain NEM date and lead band."""
    return f"{retrain}/{band}"


# --------------------------------------------------------------------------- schedule


def add_months(day: dt.date, months: int) -> dt.date:
    """The same day ``months`` later (clipped to the month's end)."""
    return (pd.Timestamp(day) + pd.DateOffset(months=months)).date()


def retrain_schedule(first: dt.date, end: pd.Timestamp, every_months: int) -> list[pd.Timestamp]:
    """NEM-midnight retrain instants ``first``, ``first + k months``, … strictly before ``end``."""
    out: list[pd.Timestamp] = []
    day = first
    while (ts := nem_midnight_utc(day.isoformat())) < end:
        out.append(ts)
        day = add_months(day, every_months)
    return out


# --------------------------------------------------------------------------- samples


def epoch_steps(times: pd.DatetimeIndex) -> np.ndarray:
    """Whole 5-minute steps since the Unix epoch (unit-agnostic)."""
    return ((times - pd.Timestamp(0, tz="UTC")) // INTERVAL).to_numpy(dtype=np.int64)


def stagger_phase(times: pd.DatetimeIndex, setup: ForecastSetup, every: int) -> np.ndarray:
    """Phase 0…every−1 of each issue time; the sample uses horizons ``phase + 1 + every·j``.

    The phase advances by one per issue time and by one more each UTC day, so a given
    issue time of day cycles through every phase over ``every`` days (with 48 issue times
    a day and ``every = 3``, the per-issue-time rotation alone would repeat daily).
    """
    steps = epoch_steps(times)
    if np.any(steps % setup.issue_every_steps):
        raise ValueError("issue times must lie on the issue grid")
    k = steps // setup.issue_every_steps
    per_day = DAY_STEPS // setup.issue_every_steps
    return np.asarray((k + k // per_day) % every, dtype=np.int64)


@dataclass(slots=True)
class Sample:
    """Feature rows of one variant for (issue time, horizon) pairs, stored as float32."""

    variant: str
    columns: list[str]
    origins: np.ndarray  # int64 origin position per row
    horizons: np.ndarray  # int16
    X: np.ndarray  # float32 (n, p)
    actual: np.ndarray  # float64 (NaN where missing)
    complete: np.ndarray  # bool: every feature present

    @property
    def targets(self) -> np.ndarray:
        return self.origins + self.horizons

    @classmethod
    def from_matrix(cls, fm: FeatureMatrix) -> Sample:
        X = fm.X.to_numpy(dtype=np.float32, na_value=np.nan)
        return cls(
            variant=fm.variant,
            columns=list(fm.X.columns),
            origins=fm.origins.astype(np.int64),
            horizons=fm.horizons.astype(np.int16),
            X=X,
            actual=fm.actual.astype(np.float64),
            complete=np.isfinite(X).all(axis=1),
        )

    @classmethod
    def concat(cls, parts: Sequence[Sample]) -> Sample:
        if not parts:
            raise ValueError("no sample parts")
        return cls(
            variant=parts[0].variant,
            columns=parts[0].columns,
            origins=np.concatenate([p.origins for p in parts]),
            horizons=np.concatenate([p.horizons for p in parts]),
            X=np.concatenate([p.X for p in parts]),
            actual=np.concatenate([p.actual for p in parts]),
            complete=np.concatenate([p.complete for p in parts]),
        )

    @property
    def usable(self) -> np.ndarray:
        return self.complete & np.isfinite(self.actual)


def _month_chunks(builder: FeatureBuilder, origins: np.ndarray) -> Iterable[np.ndarray]:
    for _, idx in monthly_folds(pd.DatetimeIndex(builder.ctx.times[origins])):
        yield origins[idx]


def training_sample(
    builder: FeatureBuilder, variant: str, origins: np.ndarray, every: int
) -> Sample:
    """Staggered training rows of ``variant`` for ``origins`` (built month by month)."""
    setup = builder.setup
    if setup.horizon_steps % every:
        raise ValueError("sample_every_steps must divide the number of horizons")
    parts: list[Sample] = []
    for chunk in _month_chunks(builder, origins):
        phase = stagger_phase(pd.DatetimeIndex(builder.ctx.times[chunk]), setup, every)
        for p in range(every):
            sel = chunk[phase == p]
            if sel.size:
                horizons = np.arange(p + 1, setup.horizon_steps + 1, every)
                parts.append(Sample.from_matrix(builder.build(variant, sel, horizons)))
    return Sample.concat(parts)


def shared_usable(samples: Sequence[Sample]) -> np.ndarray:
    """Rows every sample can use; the samples must describe the same pairs, in order."""
    first = samples[0]
    mask = first.usable.copy()
    for s in samples[1:]:
        if not (
            np.array_equal(s.origins, first.origins) and np.array_equal(s.horizons, first.horizons)
        ):
            raise ValueError("samples of one group must cover identical pairs")
        mask &= s.usable
    return mask


def fit_rows(
    sample: Sample, usable: np.ndarray, cutoff: int, es_steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training rows at ``cutoff`` and the early-stopping split within them.

    Returns ``(rows, es_train, es_valid)``: row indices with a known target
    (``t <= cutoff``) and two masks over those rows. ``es_valid`` holds the pairs issued in
    the last ``es_steps`` before the cutoff; ``es_train`` the pairs whose target precedes
    that slice, so the monitored pairs come strictly after everything fitted.
    """
    targets = sample.targets
    rows = np.flatnonzero(usable & (targets <= cutoff))
    split = cutoff - es_steps
    es_valid = sample.origins[rows] >= split
    es_train = targets[rows] <= split
    return rows, es_train, es_valid


# --------------------------------------------------------------------------- walk-forward


@dataclass(slots=True)
class RetrainRecord:
    retrain: str  # NEM date of the retrain
    band: str  # lead band of this fit (band_key)
    cutoff: str  # last known target (UTC)
    fold_origins: int
    rows: int  # training rows of the final fit (before ridge's subsample)
    es_train_rows: int
    es_valid_rows: int
    chosen: dict[str, Any]  # winning setting
    params: dict[str, Any]  # final model's parameters (incl. LightGBM tree count)
    candidates: list[dict[str, Any]]  # every setting scored: setting + Score fields
    refit_seconds: float
    path: str | None  # model file, relative to the data root


@dataclass(slots=True)
class WalkForwardResult:
    origins: np.ndarray  # evaluation origin positions
    predictions: dict[str, np.ndarray]  # model name -> (n_origins, H) float32
    folds: list[str]  # retrain NEM dates with at least one issue time
    retrains: dict[str, list[RetrainRecord]] = field(default_factory=dict)
    train_rows: dict[str, int] = field(default_factory=dict)  # rows of the group's sample
    usable_rows: int = 0  # rows every variant of the group can use


@dataclass(slots=True)
class BandedModel:
    """One fitted model per lead band; each forecast row goes to its horizon's band."""

    bands: dict[tuple[int, int], MlModel] = field(default_factory=dict)

    def predict(self, X: np.ndarray, horizons: np.ndarray) -> np.ndarray:
        out = np.full(len(X), np.nan)
        covered = np.zeros(len(X), dtype=bool)
        for (lo, hi), model in self.bands.items():
            sel = (horizons >= lo) & (horizons <= hi)
            covered |= sel
            if sel.any():
                out[sel] = model.predict(X[sel])
        if not covered.all():
            raise ValueError("a horizon lies outside every lead band")
        return out


def predict_origins(
    builder: FeatureBuilder,
    variant: str,
    origins: np.ndarray,
    models: Mapping[str, BandedModel],
) -> dict[str, np.ndarray]:
    """``(n_origins, H)`` forecasts of each model; NaN where a feature is missing."""
    H = builder.setup.horizon_steps
    out = {name: np.full((len(origins), H), np.nan, np.float32) for name in models}
    start = 0
    for chunk in _month_chunks(builder, origins):
        sample = Sample.from_matrix(builder.build(variant, chunk))
        rows = np.flatnonzero(sample.complete)
        for name, model in models.items():
            flat = np.full(len(sample.X), np.nan, np.float32)
            if rows.size:
                flat[rows] = model.predict(sample.X[rows], sample.horizons[rows])
            out[name][start : start + len(chunk)] = flat.reshape(len(chunk), H)
        start += len(chunk)
    return out


def grid_position(builder: FeatureBuilder, ts: pd.Timestamp) -> int:
    """Position of ``ts`` on the 5-minute data grid (it must lie on it)."""
    pos = int(builder.ctx.times.searchsorted(ts))
    if pos >= len(builder.ctx.times) or builder.ctx.times[pos] != ts:
        raise LookupError(f"{ts} is not on the data grid")
    return pos


def train_origins(
    builder: FeatureBuilder, before: int, start: pd.Timestamp | None = None
) -> np.ndarray:
    """Issue-grid positions with a full week of history, from ``start``, before ``before``."""
    times = builder.ctx.times
    steps = epoch_steps(pd.DatetimeIndex(times))
    pos = np.arange(len(times))
    ok = (steps % builder.setup.issue_every_steps == 0) & (pos >= WEEK_STEPS) & (pos < before)
    if start is not None:
        ok &= times >= start
    return pos[ok]


def select_and_refit(
    spec: ModelSpec,
    columns: list[str],
    X: np.ndarray,
    actual: np.ndarray,
    es_train: np.ndarray,
    es_valid: np.ndarray,
    key: str,
    cfg: MlConfig,
) -> tuple[MlModel, RetrainRecord]:
    """Score the spec's candidates (or take its fixed setting for ``key``) and refit the winner.

    The returned record's ``retrain``, ``band``, ``cutoff`` and ``fold_origins`` are filled
    in by the caller.
    """
    if spec.fixed is not None:
        if key not in spec.fixed:
            raise KeyError(f"{spec.name}: no fixed setting for {key}")
        candidates: tuple[Mapping[str, Any], ...] = (spec.fixed[key],)
    else:
        candidates = spec.candidates
    scores: list[Score] = []
    for choice in candidates:
        # Each candidate is fitted and dropped before the next, to bound memory.
        scores.append(
            make_model(spec.family, columns, choice, cfg).score(X, actual, es_train, es_valid)
        )
    best = min(range(len(scores)), key=lambda i: scores[i].mae)  # first on ties
    final = make_model(spec.family, columns, candidates[best], cfg)
    seconds = final.refit(X, actual, scores[best].best_iteration)
    record = RetrainRecord(
        retrain="",
        band="",
        cutoff="",
        fold_origins=0,
        rows=len(X),
        es_train_rows=int(es_train.sum()),
        es_valid_rows=int(es_valid.sum()),
        chosen=dict(candidates[best]),
        params=final.params(),
        candidates=[
            {"setting": dict(c), **sc.as_dict()} for c, sc in zip(candidates, scores, strict=True)
        ],
        refit_seconds=seconds,
        path=None,
    )
    return final, record


def run_walk_forward(
    builder: FeatureBuilder,
    specs: Sequence[ModelSpec],
    eval_origins: np.ndarray,
    retrains: Sequence[pd.Timestamp],
    cfg: MlConfig,
    train_start: pd.Timestamp | None = None,
    model_dir: Path | None = None,
    data_root: Path | None = None,
) -> WalkForwardResult:
    """Fit every spec at each retrain and forecast the issue times of its fold."""
    if not specs or not retrains:
        raise ValueError("need at least one model and one retrain")
    ctx, setup = builder.ctx, builder.setup
    eval_times = pd.DatetimeIndex(ctx.times[eval_origins])
    if eval_times.size == 0 or eval_times[0] < retrains[0]:
        raise ValueError("every evaluation issue time must follow the first retrain")
    variants = list(dict.fromkeys(s.variant for s in specs))
    cutoffs = [grid_position(builder, r) - setup.latency_steps for r in retrains]
    es_steps = cfg.early_stopping_weeks * 7 * DAY_STEPS

    origins = train_origins(builder, cutoffs[-1], train_start)
    samples = {v: training_sample(builder, v, origins, cfg.sample_every_steps) for v in variants}
    usable = shared_usable(list(samples.values()))
    log.info(
        "walkforward.sample",
        variants=variants,
        rows=len(usable),
        usable=int(usable.sum()),
        first_issue=str(ctx.times[origins[0]]),
    )
    result = WalkForwardResult(
        origins=eval_origins,
        predictions={
            s.name: np.full((len(eval_origins), setup.horizon_steps), np.nan, np.float32)
            for s in specs
        },
        folds=[],
        retrains={s.name: [] for s in specs},
        train_rows={v: len(s.X) for v, s in samples.items()},
        usable_rows=int(usable.sum()),
    )
    for i, (retrain, cutoff) in enumerate(zip(retrains, cutoffs, strict=True)):
        in_fold = eval_times >= retrain
        if i + 1 < len(retrains):
            in_fold &= eval_times < retrains[i + 1]
        if not in_fold.any():
            continue
        label = retrain.tz_convert(NEM_TZ).strftime("%Y-%m-%d")
        result.folds.append(label)
        fold = eval_origins[in_fold]
        for variant in variants:
            sample = samples[variant]
            rows, es_train, es_valid = fit_rows(sample, usable, cutoff, es_steps)
            horizons = sample.horizons[rows]
            variant_specs = [s for s in specs if s.variant == variant]
            fitted = {spec.name: BandedModel() for spec in variant_specs}
            for band in cfg.lead_bands:
                bkey = band_key(band)
                in_band = (horizons >= band[0]) & (horizons <= band[1])
                X, actual = sample.X[rows[in_band]], sample.actual[rows[in_band]]
                for spec in variant_specs:
                    model, record = select_and_refit(
                        spec,
                        sample.columns,
                        X,
                        actual,
                        es_train[in_band],
                        es_valid[in_band],
                        fit_key(label, bkey),
                        cfg,
                    )
                    record.retrain, record.band = label, bkey
                    record.cutoff = str(ctx.times[cutoff])
                    record.fold_origins = len(fold)
                    if model_dir is not None:
                        written = model.save(model_dir / spec.name / f"{label}_{bkey}")
                        record.path = str(written.relative_to(data_root) if data_root else written)
                    result.retrains[spec.name].append(record)
                    fitted[spec.name].bands[band] = model
                    log.info(
                        "walkforward.fit",
                        model=spec.name,
                        retrain=label,
                        band=bkey,
                        chosen=record.chosen,
                        es_mae=next(
                            (c["mae"] for c in record.candidates if c["setting"] == record.chosen),
                            None,
                        ),
                    )
                del X, actual
            for name, pred in predict_origins(builder, variant, fold, fitted).items():
                result.predictions[name][in_fold] = pred
            log.info("walkforward.fold", variant=variant, retrain=label, origins=len(fold))
    return result
