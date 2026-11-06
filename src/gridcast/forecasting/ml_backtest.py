"""The M2 ML walk-forward run: fit, forecast, rescore baselines on identical pairs, store.

Three evaluation sets, each stored as one batch of ``model_runs`` rows
(``config.evaluation_set``):

* ``main``: LightGBM and ridge for A, C and D plus the M1 demand baselines, on every
  development issue time from ``ml.first_issue``. Every model is scored on the pairs that
  every model can score, so pairs without C weather are excluded for all of them
  (m2_plan.md §2.4). S1–S4 are judged on this set (step 4).
* ``plus``: C⁺ and D⁺ (issue times from ``ml.plus_first_issue``), compared with C and the
  baselines on their common pairs. Reported, not pass/fail.
* ``delay``: C rebuilt with the longer GFS publication delay
  (``features.delay_sensitivity_hours``), using headline C's per-retrain settings, compared
  with headline C on their common pairs. A robustness check, never the headline.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from gridcast.config import AppConfig
from gridcast.evaluation.segments import weather_segments
from gridcast.evaluation.store import StoredModel, save_predictions, store_backtest
from gridcast.features.builder import FeatureBuilder
from gridcast.features.inputs import load_context, load_feature_builder
from gridcast.forecasting.backtest import BacktestOutput, evaluate
from gridcast.forecasting.origins import nem_midnight_utc, origin_positions, target_positions
from gridcast.forecasting.walkforward import (
    ModelSpec,
    WalkForwardResult,
    fit_key,
    model_name,
    retrain_schedule,
    run_walk_forward,
)
from gridcast.logs import get_logger
from gridcast.models.baselines import default_baselines
from gridcast.models.ml import FAMILIES, grid_choices

log = get_logger(__name__)

MAIN_VARIANTS = ("A", "C", "D")
PLUS_VARIANTS = ("Cplus", "Dplus")
TARGET = "demand"


@dataclass(slots=True)
class MlRecord:
    """A walk-forward ML model as stored in ``model_runs`` (satisfies ``StoredModel``)."""

    name: str
    details: dict[str, Any]
    kind: str = "ml"

    def params(self) -> dict[str, object]:
        return self.details


@dataclass(slots=True)
class EvaluationSet:
    key: str
    out: BacktestOutput
    metrics: pd.DataFrame
    models: list[StoredModel]
    run_config: dict[str, Any]
    batch_id: uuid.UUID = field(default_factory=uuid.uuid4)


def ml_config_hash(cfg: AppConfig) -> str:
    """Hash of every setting that changes the ML forecasts (not paths or reporting)."""
    payload = {
        "ml": cfg.ml.model_dump(mode="json", exclude={"model_dir"}),
        "features": cfg.features.model_dump(mode="json"),
        "forecast": cfg.forecast.model_dump(mode="json"),
        "weather_forecasts": cfg.weather.forecasts.model_dump(
            mode="json",
            include={"model", "offsets_days", "run_every_hours", "publication_delay_hours"},
        ),
        "weather_observations": cfg.weather.observations.model_dump(mode="json"),
        "sites": {k: v.model_dump(mode="json") for k, v in cfg.weather.sites.items()},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def grid_specs(variants: Sequence[str], cfg: AppConfig) -> list[ModelSpec]:
    return [
        ModelSpec(model_name(f, v), v, f, candidates=tuple(grid_choices(f, cfg.ml)))
        for v in variants
        for f in FAMILIES
    ]


def follow_specs(variant: str, suffix: str, headline: WalkForwardResult) -> list[ModelSpec]:
    """Specs that reuse the settings headline ``variant`` chose at each retrain."""
    specs = []
    for family in FAMILIES:
        source = model_name(family, variant)
        fixed = {fit_key(r.retrain, r.band): r.chosen for r in headline.retrains[source]}
        specs.append(ModelSpec(model_name(family, variant, suffix), variant, family, fixed=fixed))
    return specs


def _records(result: WalkForwardResult, specs: Sequence[ModelSpec]) -> list[MlRecord]:
    out = []
    for spec in specs:
        out.append(
            MlRecord(
                spec.name,
                {
                    "family": spec.family,
                    "variant": spec.variant,
                    "selection": "fixed" if spec.fixed is not None else "per_retrain_grid",
                    "grid": [dict(c) for c in spec.candidates],
                    "retrains": [asdict(r) for r in result.retrains[spec.name]],
                },
            )
        )
    return out


def _baseline_predictions(builder: FeatureBuilder, origins: np.ndarray) -> dict[str, np.ndarray]:
    return {
        m.name: m.predict(builder.ctx, origins, builder.setup).astype(np.float32)
        for m in default_baselines(TARGET)
    }


def _output(
    builder: FeatureBuilder,
    origins: np.ndarray,
    predictions: dict[str, np.ndarray],
    folds: list[str],
) -> BacktestOutput:
    targets = target_positions(origins, builder.setup)
    return BacktestOutput(
        target=TARGET,
        setup=builder.setup,
        origins=origins,
        origin_times=pd.DatetimeIndex(builder.ctx.times[origins]),
        targets=targets,
        actual=builder.ctx.y[targets],
        predictions=predictions,
        folds=folds,
    )


def run_ml_backtest(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    until: pd.Timestamp | None = None,
    plus: bool = True,
    delay: bool = True,
    model_root: Path | None = None,
    on_set: Callable[[EvaluationSet], None] | None = None,
) -> list[EvaluationSet]:
    """Run the walk-forward and return the evaluation sets (stored by the caller).

    ``until`` ends the evaluation early (a smoke run); ``model_root`` (normally
    ``<data root>/models``) receives one directory per batch with every fitted model.
    """
    ctx = load_context(engine, cfg, region_id)
    builder = load_feature_builder(engine, cfg, region_id, ctx=ctx)
    setup = builder.setup
    first = nem_midnight_utc(str(cfg.ml.first_issue))
    holdout = nem_midnight_utc(str(cfg.backtest.holdout_start))
    end = min(holdout, until) if until is not None else holdout
    origins = origin_positions(ctx, setup, first)
    origins = origins[ctx.times[origins] < end]
    if origins.size == 0:
        raise LookupError("no development issue times to evaluate")
    schedule = retrain_schedule(cfg.ml.first_issue, end, cfg.ml.retrain_every_months)
    config_hash = ml_config_hash(cfg)
    data_root = cfg.data_root() if model_root is not None else None

    assert builder.weather is not None and builder.weather.observed is not None
    observed = builder.weather.observed.series
    segments = weather_segments(
        ctx.local_day,
        ctx.frame["local_hour"].to_numpy(),
        observed["temp_weighted"],
        observed["cloud_weighted"],
        cfg.evaluation,
    )

    def finish(
        key: str,
        out: BacktestOutput,
        models: list[StoredModel],
        extra: dict[str, Any],
        batch: uuid.UUID,
    ) -> EvaluationSet:
        metrics = evaluate(out, ctx, cfg.evaluation, holdout, segments, splits=("dev",))
        mask = out.common_mask()
        run_config = {
            "evaluation_set": key,
            "config_hash": config_hash,
            "setup": {
                "issue_every_steps": setup.issue_every_steps,
                "horizon_steps": setup.horizon_steps,
                "latency_steps": setup.latency_steps,
            },
            "first_issue": str(out.origin_times[0]),
            "last_issue": str(out.origin_times[-1]),
            "complete_window": until is None,
            "ml": cfg.ml.model_dump(mode="json"),
            "lead_bands": [list(b) for b in cfg.ml.lead_bands],
            "publication_delay_hours": cfg.weather.forecasts.publication_delay_hours,
            "n_origins": len(out.origins),
            "retrains": out.folds,
            "scored_pairs": int(mask.sum()),
            "total_pairs": int(mask.size),
            **extra,
        }
        es = EvaluationSet(key, out, metrics, models, run_config, batch)
        if on_set is not None:
            on_set(es)
        return es

    def model_dir(batch: uuid.UUID) -> Path | None:
        return model_root / str(batch) if model_root is not None else None

    sets: list[EvaluationSet] = []

    # ---------------------------------------------------------------- main: A, C, D
    batch = uuid.uuid4()
    specs = grid_specs(MAIN_VARIANTS, cfg)
    main = run_walk_forward(
        builder, specs, origins, schedule, cfg.ml, model_dir=model_dir(batch), data_root=data_root
    )
    baselines = _baseline_predictions(builder, origins)
    out = _output(builder, origins, {**main.predictions, **baselines}, main.folds)
    models: list[StoredModel] = [*_records(main, specs), *default_baselines(TARGET)]
    extra = {"train_rows": main.train_rows, "shared_train_rows": main.usable_rows}
    sets.append(finish("main", out, models, extra, batch))

    # ---------------------------------------------------------------- plus: C⁺, D⁺
    plus_first = nem_midnight_utc(str(cfg.ml.plus_first_issue))
    plus_origins = origins[ctx.times[origins] >= plus_first]
    if plus and plus_origins.size:
        batch = uuid.uuid4()
        plus_specs = grid_specs(PLUS_VARIANTS, cfg)
        plus_schedule = [r for r in schedule if r >= plus_first]
        if not plus_schedule or plus_schedule[0] != plus_first:
            raise ValueError("ml.plus_first_issue must be one of the retrain dates")
        res = run_walk_forward(
            builder,
            plus_specs,
            plus_origins,
            plus_schedule,
            cfg.ml,
            train_start=nem_midnight_utc(str(cfg.ml.plus_train_start)),
            model_dir=model_dir(batch),
            data_root=data_root,
        )
        sel = np.isin(origins, plus_origins)
        reference = {
            name: main.predictions[name][sel] for name in (model_name(f, "C") for f in FAMILIES)
        }
        base_plus = {k: v[sel] for k, v in baselines.items()}
        out = _output(
            builder, plus_origins, {**res.predictions, **reference, **base_plus}, res.folds
        )
        c_records = [r for r in _records(main, specs) if r.name in reference]
        models = [*_records(res, plus_specs), *c_records, *default_baselines(TARGET)]
        extra = {"train_rows": res.train_rows, "shared_train_rows": res.usable_rows}
        sets.append(finish("plus", out, models, extra, batch))

    # ---------------------------------------------------------------- delay: C, later GFS
    if delay:
        hours = cfg.features.delay_sensitivity_hours
        batch = uuid.uuid4()
        delayed = load_feature_builder(engine, cfg, region_id, delay_hours=hours, ctx=ctx)
        suffix = f"_delay{hours:g}h"
        delay_specs = follow_specs("C", suffix, main)
        res = run_walk_forward(
            delayed,
            delay_specs,
            origins,
            schedule,
            cfg.ml,
            model_dir=model_dir(batch),
            data_root=data_root,
        )
        reference = {
            name: main.predictions[name] for name in (model_name(f, "C") for f in FAMILIES)
        }
        out = _output(builder, origins, {**reference, **res.predictions}, res.folds)
        c_records = [r for r in _records(main, specs) if r.name in reference]
        models = [*c_records, *_records(res, delay_specs)]
        extra = {
            "delay_hours": hours,
            "train_rows": res.train_rows,
            "shared_train_rows": res.usable_rows,
        }
        sets.append(finish("delay", out, models, extra, batch))
    return sets


def store_sets(
    engine: Engine, cfg: AppConfig, region_id: str, sets: Sequence[EvaluationSet]
) -> dict[str, uuid.UUID]:
    """Write predictions (parquet) and ``model_runs``/``forecast_metrics`` rows per set."""
    holdout = nem_midnight_utc(str(cfg.backtest.holdout_start))
    out: dict[str, uuid.UUID] = {}
    for es in sets:
        written = save_predictions(es.out, cfg.data_path(cfg.paths.backtest_dir), es.batch_id)
        artifacts = {name: path.relative_to(cfg.data_root()) for name, path in written.items()}
        store_backtest(
            engine,
            es.out,
            es.metrics,
            es.models,
            region_id,
            holdout,
            es.run_config,
            artifacts,
            es.batch_id,
        )
        out[es.key] = es.batch_id
    return out
