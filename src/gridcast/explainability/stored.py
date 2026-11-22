"""The stored M2 LightGBM C fits that SHAP explains (m4_plan.md §2.6). No model is refitted.

* **Batches:** development explains run 2 (``explain.dev_batch``, evaluation set ``main``),
  the holdout explains ``holdout_main`` (``explain.holdout_batch``). Both must be complete
  runs of the frozen M2 recipe (``backtest.frozen_ml_config_hash``).
* **Fits:** one per (retrain, lead band), as M2's ``RetrainRecord`` stored them: the chosen
  target form, objective and leaf count, the tree count, and the model file.
* **Folds:** a fit forecasts the issue times from its retrain (NEM midnight) up to the next
  retrain of the same batch, as in :func:`~gridcast.forecasting.walkforward.run_walk_forward`.
* **Loading** rebuilds M2's own :class:`~gridcast.models.ml.LightGBMModel` around the stored
  booster, so a forecast goes through M2's code path (``baseline + booster.predict``).
"""

from __future__ import annotations

import gzip
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from gridcast.config import AppConfig, LightGBMConfig
from gridcast.evaluation.holdout import HOLDOUT_SETS
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.forecasting.walkforward import band_key
from gridcast.models.ml import LightGBMModel

DEV_SET = "main"
HOLDOUT_SET = HOLDOUT_SETS["main"]


@dataclass(frozen=True, slots=True)
class StoredFit:
    retrain: str  # NEM date of the retrain, as M2 recorded it
    band: tuple[int, int]  # lead band (steps), inclusive
    target_form: str
    objective: str
    num_leaves: int
    n_trees: int
    path: Path  # absolute model file
    fold_start: pd.Timestamp  # UTC: first issue time this fit forecasts
    fold_end: pd.Timestamp | None  # UTC, exclusive: the batch's next retrain (None: last)

    @property
    def key(self) -> str:
        return f"{self.retrain}_{band_key(self.band)}"

    def covers(self, issue_times: pd.DatetimeIndex, leads: np.ndarray) -> np.ndarray:
        inside = np.asarray(issue_times >= self.fold_start)
        if self.fold_end is not None:
            inside &= np.asarray(issue_times < self.fold_end)
        leads = np.asarray(leads)
        return inside & (leads >= self.band[0]) & (leads <= self.band[1])


def parse_band(key: str) -> tuple[int, int]:
    """``h007-024`` → (7, 24)."""
    if len(key) != 8 or key[0] != "h" or key[4] != "-":
        raise ValueError(f"not a lead-band key: {key!r}")
    band = (int(key[1:4]), int(key[5:8]))
    if band_key(band) != key:
        raise ValueError(f"not a lead-band key: {key!r}")
    return band


def fits_from_records(
    records: Sequence[dict[str, Any]],
    data_root: Path,
    lead_bands: Sequence[Sequence[int]],
    period: tuple[pd.Timestamp, pd.Timestamp | None] | None = None,
) -> list[StoredFit]:
    """Fits from M2's stored retrain records, checked to cover every band at every retrain.

    ``period`` (first issue time, exclusive end) clips every fold to the batch's own issue
    times. ``holdout_main``'s first retrain (2025-08-01) also forecast Aug–Sep 2025, which
    are development issue times explained by run 2's identical fit, so its fold starts at
    the holdout.
    """
    bands = [tuple(int(v) for v in b) for b in lead_bands]
    retrains = sorted({str(r["retrain"]) for r in records})
    starts = {d: nem_midnight_utc(d) for d in retrains}
    out = []
    for r in records:
        band = parse_band(str(r["band"]))
        if band not in bands:
            raise ValueError(f"record band {r['band']} is not one of ml.lead_bands")
        chosen, params = r["chosen"], r["params"]
        if r.get("path") is None:
            raise ValueError(f"fit {r['retrain']} {r['band']} has no stored model file")
        i = retrains.index(str(r["retrain"]))
        start = starts[retrains[i]]
        end = starts[retrains[i + 1]] if i + 1 < len(retrains) else None
        if period is not None:
            start = max(start, period[0])
            if period[1] is not None:
                end = period[1] if end is None else min(end, period[1])
        out.append(
            StoredFit(
                retrain=str(r["retrain"]),
                band=(band[0], band[1]),
                target_form=str(chosen["target_form"]),
                objective=str(chosen["objective"]),
                num_leaves=int(chosen["num_leaves"]),
                n_trees=int(params["n_trees"]),
                path=data_root / str(r["path"]),
                fold_start=start,
                fold_end=end,
            )
        )
    seen = {(f.retrain, f.band) for f in out}
    if len(seen) != len(out) or seen != {(d, b) for d in retrains for b in bands}:
        raise ValueError("the records must hold exactly one fit per retrain and lead band")
    return sorted(out, key=lambda f: (f.retrain, f.band))


def batch_config(
    engine: Engine, cfg: AppConfig, batch_id: str, evaluation_set: str
) -> dict[str, Any]:
    """The stored config of the batch's explained model, checked against the frozen recipe."""
    model = cfg.anomalies.explain.model
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT config FROM model_runs
                WHERE batch_id = CAST(:b AS uuid) AND model_name = :m AND status = 'success'
                """
            ),
            {"b": batch_id, "m": model},
        ).one_or_none()
    if row is None:
        raise LookupError(f"batch {batch_id} has no successful {model} run")
    config: dict[str, Any] = row[0]
    problems = []
    if config.get("evaluation_set") != evaluation_set:
        problems.append(f"evaluation set {config.get('evaluation_set')!r}, not {evaluation_set!r}")
    if not config.get("complete_window"):
        problems.append("not a complete window")
    if config.get("config_hash") != cfg.backtest.frozen_ml_config_hash:
        problems.append(
            f"ML config hash {config.get('config_hash')}, not the frozen "
            f"{cfg.backtest.frozen_ml_config_hash}"
        )
    if problems:
        raise ValueError(f"batch {batch_id}: " + "; ".join(problems))
    return config


def split_period(cfg: AppConfig, split: str) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    """Issue times a split explains: development [ml.first_issue, holdout_start), holdout
    [holdout_start, holdout_end)."""
    holdout = nem_midnight_utc(str(cfg.backtest.holdout_start))
    if split == "dev":
        return nem_midnight_utc(str(cfg.ml.first_issue)), holdout
    if split == "holdout":
        end = cfg.backtest.holdout_end
        return holdout, nem_midnight_utc(str(end)) if end is not None else None
    raise ValueError(f"unknown split {split!r}")


def stored_fits(engine: Engine, cfg: AppConfig, split: str) -> list[StoredFit]:
    """Every fit of the batch that ``split`` ("dev" or "holdout") explains, folds clipped to
    the split's issue times."""
    ex = cfg.anomalies.explain
    period = split_period(cfg, split)
    batch, evaluation_set = (
        (ex.dev_batch, DEV_SET) if split == "dev" else (ex.holdout_batch, HOLDOUT_SET)
    )
    config = batch_config(engine, cfg, batch, evaluation_set)
    return fits_from_records(
        config["model_params"]["retrains"], cfg.data_root(), cfg.ml.lead_bands, period
    )


def load_model(
    fit: StoredFit, columns: Sequence[str], lgb_cfg: LightGBMConfig, seed: int
) -> LightGBMModel:
    """M2's model object around the stored booster; refuses a file that does not match."""
    import lightgbm as lgb

    with gzip.open(fit.path, "rt", encoding="utf-8") as fh:
        booster = lgb.Booster(model_str=fh.read())
    if list(booster.feature_name()) != list(columns):
        raise ValueError(f"{fit.path.name}: feature columns differ from the variant's")
    if booster.num_trees() != fit.n_trees:
        raise ValueError(
            f"{fit.path.name}: {booster.num_trees()} trees, the record says {fit.n_trees}"
        )
    model = LightGBMModel(columns, fit.target_form, fit.objective, fit.num_leaves, lgb_cfg, seed)
    model.booster = booster
    model.n_trees = fit.n_trees
    return model


def locate(
    fits: Sequence[StoredFit], issue_times: pd.DatetimeIndex, leads: np.ndarray
) -> np.ndarray:
    """Per pair, the position in ``fits`` of the fit that forecast it (−1: none)."""
    out = np.full(len(issue_times), -1, dtype=np.int64)
    for k, fit in enumerate(fits):
        hit = fit.covers(issue_times, leads)
        if np.any(hit & (out >= 0)):
            raise ValueError("two fits cover the same pair")
        out[hit] = k
    return out
