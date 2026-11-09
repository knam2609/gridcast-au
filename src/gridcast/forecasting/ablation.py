"""The weather-feature drop/add test of m2_plan.md §2.5 ("investigate before believing").

The plan says any feature that improves pooled MAE by more than 15% on its own, with one
feature added or removed at a time, is checked for leakage first. A full walk-forward per
feature would take several overnight runs, so (agreed for step 4, 2026-11-06) the test
runs at a few retrains (``evaluation.weather_ablation_retrains``) on their early-stopping
slices:

* the training rows, early-stopping split and lead bands are exactly those of the stored
  ``main`` run (same sample, same shared rows, same cutoff);
* every fit uses the setting headline LightGBM C chose at that retrain and band, so only
  the features change;
* feature sets: all C features (``full``, which must reproduce the stored early-stopping
  MAE), no weather (``none``), and every weather feature dropped from C (``drop``) or
  added alone to the weather-free set (``add``).

A feature is flagged when adding it alone, or keeping it among the others, cuts the
early-stopping MAE (pooled over the bands, weighted by their monitored rows) by more than
``evaluation.ablation_flag_share``. Flags call for the leakage evidence in the M2 report;
they are not a verdict.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from gridcast.config import AppConfig
from gridcast.evaluation.m2_results import load_evaluation_set, retrain_records, run_config
from gridcast.features import catalogue
from gridcast.features.inputs import load_feature_builder
from gridcast.forecasting.ml_backtest import ml_config_hash
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.forecasting.walkforward import (
    band_key,
    fit_rows,
    grid_position,
    shared_usable,
    train_origins,
    training_sample,
)
from gridcast.logs import get_logger
from gridcast.models.baselines import DAY_STEPS
from gridcast.models.ml import make_model
from gridcast.provenance import git_sha

log = get_logger(__name__)

HEADLINE = "lgbm_C"
VARIANT = "C"
WEATHER_FREE = "A"
ABLATION_DIR = "ablations"


@dataclass(slots=True)
class AblationFit:
    retrain: str
    band: str
    kind: str  # full | none | drop | add
    feature: str | None  # the weather feature dropped or added
    n_features: int
    es_mae: float
    es_valid_rows: int
    best_iteration: int | None
    seconds: float


def feature_sets(columns: list[str], weather: list[str]) -> list[tuple[str, str | None, list[str]]]:
    """``(kind, feature, columns)`` for every fit, columns kept in the variant's order."""
    missing = set(weather) - set(columns)
    if missing:
        raise ValueError(f"weather features not in the columns: {sorted(missing)}")
    base = [c for c in columns if c not in weather]
    sets: list[tuple[str, str | None, list[str]]] = [("full", None, columns), ("none", None, base)]
    sets += [("drop", f, [c for c in columns if c != f]) for f in weather]
    sets += [("add", f, [c for c in columns if c not in weather or c == f]) for f in weather]
    return sets


def pooled_mae(fits: pd.DataFrame) -> pd.DataFrame:
    """Early-stopping MAE pooled over lead bands (weighted by monitored rows).

    One row per (retrain, kind, feature); ``feature`` is ``""`` for ``full`` and ``none``.
    """
    df = fits.assign(feature=fits["feature"].fillna(""), w=fits["es_mae"] * fits["es_valid_rows"])
    g = df.groupby(["retrain", "kind", "feature"], sort=False)
    out = (g["w"].sum() / g["es_valid_rows"].sum()).rename("es_mae").reset_index()
    return out


def flag_table(fits: pd.DataFrame, share: float) -> pd.DataFrame:
    """Per retrain and weather feature: the MAE gain of adding it alone and of keeping it.

    ``add_gain`` = 1 − MAE(no weather + f) / MAE(no weather);
    ``keep_gain`` = 1 − MAE(all C) / MAE(all C without f);
    ``group_gain`` = 1 − MAE(all C) / MAE(no weather), the same for every feature.
    """
    p = pooled_mae(fits).set_index(["retrain", "kind", "feature"])["es_mae"]
    rows = []
    for retrain in dict.fromkeys(fits["retrain"]):
        full, none = p[(retrain, "full", "")], p[(retrain, "none", "")]
        weather = [f for (r, k, f) in p.index if r == retrain and k == "add"]
        for f in weather:
            add_gain = 1 - p[(retrain, "add", f)] / none
            keep_gain = 1 - full / p[(retrain, "drop", f)]
            rows.append(
                {
                    "retrain": retrain,
                    "feature": f,
                    "mae_none": none,
                    "mae_add": p[(retrain, "add", f)],
                    "mae_drop": p[(retrain, "drop", f)],
                    "mae_full": full,
                    "add_gain": add_gain,
                    "keep_gain": keep_gain,
                    "group_gain": 1 - full / none,
                    "flagged": bool(add_gain > share or keep_gain > share),
                }
            )
    return pd.DataFrame(rows)


def ablation_path(cfg: AppConfig, batch_id: str) -> Path:
    return cfg.data_root() / ABLATION_DIR / f"weather_{batch_id}.json"


def run_weather_ablation(engine: Engine, cfg: AppConfig, region_id: str) -> dict[str, Any]:
    """Run the test against the stored ``main`` batch and return the JSON-ready result."""
    sha = git_sha()  # the code that runs is the code loaded now
    retrains = [d.isoformat() for d in cfg.evaluation.weather_ablation_retrains]
    if not retrains:
        raise ValueError("evaluation.weather_ablation_retrains is empty")
    main = load_evaluation_set(engine, region_id, "main")
    stored = run_config(main, HEADLINE)
    config_hash = ml_config_hash(cfg)
    if stored["config_hash"] != config_hash:
        raise ValueError(
            f"the ML config changed since batch {main.batch_id} "
            f"({stored['config_hash']} → {config_hash}); re-run `gridcast backtest-ml` first"
        )
    schedule = list(stored["retrains"])
    unknown = sorted(set(retrains) - set(schedule))
    if unknown:
        raise ValueError(f"not retrains of batch {main.batch_id}: {unknown}")
    records = {(r["retrain"], r["band"]): r for r in retrain_records(main, HEADLINE)}

    builder = load_feature_builder(engine, cfg, region_id)
    latency = builder.setup.latency_steps
    # The stored run built its sample up to the last retrain's cutoff, shared over A/C/D.
    last_cutoff = grid_position(builder, nem_midnight_utc(schedule[-1])) - latency
    origins = train_origins(builder, last_cutoff)
    every = cfg.ml.sample_every_steps
    sample = training_sample(builder, VARIANT, origins, every)
    oracle = training_sample(builder, "D", origins, every)
    usable = shared_usable([sample, oracle])  # A's features are a subset of C's
    del oracle
    if not set(catalogue.feature_names(WEATHER_FREE)) <= set(sample.columns):
        raise ValueError("variant A must be variant C without its weather features")
    weather = [c for c in sample.columns if c not in catalogue.feature_names(WEATHER_FREE)]
    sets = feature_sets(sample.columns, weather)
    es_steps = cfg.ml.early_stopping_weeks * 7 * DAY_STEPS

    fits: list[AblationFit] = []
    reproduced: list[dict[str, Any]] = []
    for retrain in retrains:
        cutoff = grid_position(builder, nem_midnight_utc(retrain)) - latency
        rows, es_train, es_valid = fit_rows(sample, usable, cutoff, es_steps)
        horizons = sample.horizons[rows]
        for band in cfg.ml.lead_bands:
            bkey = band_key(band)
            record = records[(retrain, bkey)]
            in_band = (horizons >= band[0]) & (horizons <= band[1])
            X, actual = sample.X[rows[in_band]], sample.actual[rows[in_band]]
            tr, va = es_train[in_band], es_valid[in_band]
            for kind, feature, cols in sets:
                idx = [sample.columns.index(c) for c in cols]
                model = make_model("lightgbm", cols, record["chosen"], cfg.ml)
                score = model.score(X[:, idx], actual, tr, va)
                fits.append(
                    AblationFit(
                        retrain,
                        bkey,
                        kind,
                        feature,
                        len(cols),
                        score.mae,
                        int(va.sum()),
                        score.best_iteration,
                        score.seconds,
                    )
                )
                log.info(
                    "ablation.fit",
                    retrain=retrain,
                    band=bkey,
                    kind=kind,
                    feature=feature,
                    es_mae=round(score.mae, 2),
                    seconds=score.seconds,
                )
                if kind == "full":
                    stored_mae = next(
                        c["mae"] for c in record["candidates"] if c["setting"] == record["chosen"]
                    )
                    reproduced.append(
                        {
                            "retrain": retrain,
                            "band": bkey,
                            "stored_es_mae": stored_mae,
                            "es_mae": score.mae,
                        }
                    )
            del X, actual
    return {
        "batch_id": main.batch_id,
        "region_id": region_id,
        "config_hash": config_hash,
        "git_sha": sha,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "model": HEADLINE,
        "retrains": retrains,
        "weather_features": weather,
        "flag_share": cfg.evaluation.ablation_flag_share,
        "reproduced": reproduced,
        "fits": [asdict(f) for f in fits],
    }


def write_ablation(cfg: AppConfig, result: dict[str, Any]) -> Path:
    path = ablation_path(cfg, str(result["batch_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, default=str))
    return path


def read_ablation(cfg: AppConfig, batch_id: str) -> dict[str, Any] | None:
    path = ablation_path(cfg, batch_id)
    return json.loads(path.read_text()) if path.is_file() else None


def fits_frame(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(result["fits"])


def reproduction_ok(result: dict[str, Any], tol: float = 1e-6) -> bool:
    """Whether every ``full`` fit matched the stored run's early-stopping MAE."""
    return bool(result["reproduced"]) and all(
        np.isclose(r["es_mae"], r["stored_es_mae"], rtol=0, atol=tol) for r in result["reproduced"]
    )
