"""The global SHAP sample of a stored M2 batch (m4_plan.md §2.6; step 5, §1 "Clarified in
step 5").

``gridcast explain-dev`` draws §2.6's seeded sample (``anomalies.explain.sample_pairs`` per
fit) for every fit of run 2. It rebuilds the feature rows with M2's builder, runs TreeSHAP
on the stored booster, and checks E1's gate on every pair. It uses the frozen functions of
:mod:`~gridcast.explainability.sample` and :mod:`~gridcast.explainability.treeshap`, the
same path as the evidence's demand peaks.

* **Stored** under ``<data root>/<explain.output_dir>/<split>/``: one parquet file per fit
  (the pairs, their feature rows, actuals, stored and recomputed forecasts, anchors, base
  values and raw contributions) and ``manifest.json`` (each fit's gate, gain importance,
  eligible pairs and timings, the commit and both anomaly hashes).
* **Withheld:** if E1 fails for any pair, nothing is stored (:class:`GateError`).
* **Preconditions:** a clean tree on which both anomaly freezes hold (``explainability/``
  is under both). Reported only and deterministic, so a rerun is allowed with ``replace``.

The **whole-forecast** decomposition (user decision 2026-11-20) is :func:`whole_forecast`.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from gridcast.anomalies.freeze import freeze_state
from gridcast.config import AppConfig
from gridcast.explainability.pairs import ModelCache
from gridcast.explainability.sample import (
    Period,
    eligible_pairs,
    feature_rows,
    origin_positions,
    sample_pairs,
)
from gridcast.explainability.stored import StoredFit, split_period, stored_fits
from gridcast.explainability.treeshap import GateResult, empty_gate, explain, gate, require
from gridcast.features import catalogue
from gridcast.features.builder import FeatureBuilder
from gridcast.intervals.inputs import read_stored_batch
from gridcast.logs import get_logger
from gridcast.models.ml import BASE_FEATURE, LATEST, TODAY_VS_YESTERDAY, LightGBMModel

log = get_logger(__name__)

MANIFEST = "manifest.json"
FITS_DIR = "fits"

# Each target form's anchor (M2's ``MlModel.baseline``) as model features with weight 1.
ANCHOR_TERMS: dict[str, tuple[str, ...]] = {
    "direct": (),
    "latest": (LATEST,),
    "residual": (BASE_FEATURE,),
    "anchored": (BASE_FEATURE, TODAY_VS_YESTERDAY),
}


class GlobalRefused(RuntimeError):
    """A precondition of the global SHAP run is not met; nothing was run."""


def x_col(feature: str) -> str:
    return f"x_{feature}"


def shap_col(feature: str) -> str:
    return f"shap_{feature}"


# --------------------------------------------------------------------------- one fit


@dataclass(slots=True)
class FitSample:
    fit: StoredFit
    eligible: int  # the fit's scored pairs in the period
    frame: pd.DataFrame  # one row per sampled pair
    gate: GateResult
    gain: np.ndarray  # split gain per feature column
    seconds: dict[str, float] = field(default_factory=dict)


def anchor_from_features(frame: pd.DataFrame, target_form: str) -> np.ndarray:
    """The anchor recomputed from the stored feature rows, as M2's ``baseline`` does."""
    out = np.zeros(len(frame), dtype=np.float64)
    for term in ANCHOR_TERMS[target_form]:
        out = out + frame[x_col(term)].to_numpy(dtype=np.float64)
    return out


def explain_fit(
    fit: StoredFit,
    model: LightGBMModel,
    builder: FeatureBuilder,
    variant: str,
    issue_times: pd.DatetimeIndex,
    leads: np.ndarray,
    stored: np.ndarray,
    actual: np.ndarray,
    tolerance: float,
    eligible: int,
) -> FitSample:
    """Explain the sampled pairs of one fit; every pair must lie in its fold and band."""
    leads = np.asarray(leads, dtype=np.int64)
    if not np.all(fit.covers(issue_times, leads)):
        raise ValueError(f"fit {fit.key}: a sampled pair lies outside its fold or band")
    t0 = time.perf_counter()
    X = feature_rows(builder, variant, origin_positions(builder, issue_times), leads)
    if not np.isfinite(X).all():
        raise ValueError(f"fit {fit.key}: a sampled pair has a missing feature")
    t1 = time.perf_counter()
    ex = explain(model, X)
    result = gate(ex, stored, tolerance)
    t2 = time.perf_counter()
    columns = list(model.columns)
    data: dict[str, Any] = {
        "issue_time": issue_times,
        "lead": leads,
        "target_end": issue_times + pd.to_timedelta(leads * 5, unit="min"),
        "actual": np.asarray(actual, dtype=np.float64),
        "stored": np.asarray(stored, dtype=np.float64),
        "forecast": ex.forecast,
        "raw": ex.raw,
        "anchor": ex.anchor,
        "base": ex.base,
    }
    data |= {x_col(c): X[:, j] for j, c in enumerate(columns)}
    data |= {shap_col(c): ex.contributions[:, j] for j, c in enumerate(columns)}
    frame = pd.DataFrame(data)
    if not np.array_equal(anchor_from_features(frame, fit.target_form), ex.anchor):
        raise AssertionError(f"fit {fit.key}: the anchor is not its {fit.target_form} terms")
    if model.booster is None:
        raise RuntimeError("model is not fitted")
    gain = np.asarray(model.booster.feature_importance(importance_type="gain"), np.float64)
    return FitSample(fit, eligible, frame, result, gain, {"build": t1 - t0, "shap": t2 - t1})


# --------------------------------------------------------------------------- whole forecast


@dataclass(slots=True)
class WholeForecast:
    """forecast = constant + Σ contributions (user decision 2026-11-20, §1 step 5)."""

    columns: list[str]
    constant: np.ndarray  # (n,) base value + the sample mean of the anchor
    contributions: np.ndarray  # (n, p): TreeSHAP plus each anchor term's x − x̄


def whole_forecast(frame: pd.DataFrame, target_form: str, columns: list[str]) -> WholeForecast:
    """Credit the anchor to the features it is made of, as x − x̄ over the fit's own sample.

    That is the exact Shapley value of a linear term with the sample as background. The tree
    part stays path-dependent TreeSHAP, so the sum is the Shapley value of neither game
    exactly, but it adds up to the forecast.
    """
    contrib = frame[[shap_col(c) for c in columns]].to_numpy(dtype=np.float64).copy()
    anchor = frame["anchor"].to_numpy(dtype=np.float64)
    for term in ANCHOR_TERMS[target_form]:
        x = frame[x_col(term)].to_numpy(dtype=np.float64)
        contrib[:, columns.index(term)] += x - x.mean()
    constant = frame["base"].to_numpy(dtype=np.float64) + anchor.mean()
    return WholeForecast(list(columns), constant, contrib)


# --------------------------------------------------------------------------- the run


def output_dir(cfg: AppConfig, split: str) -> Path:
    return cfg.data_path(Path(cfg.anomalies.explain.output_dir)) / split


def check_preconditions(cfg: AppConfig, split: str, replace: bool) -> str:
    """The commit to record; refuses a dirty tree, a broken freeze or existing outputs."""
    states = [freeze_state(cfg, version=v) for v in (1, 2)]
    problems = [f"v{st.version}: {p}" for st in states for p in st.problems]
    sha = states[0].sha
    if sha is None:
        problems.append("not a git checkout")
    if any(st.dirty for st in states):
        problems.append("the working tree has uncommitted changes to tracked files")
    out = output_dir(cfg, split)
    if out.exists() and not replace:
        problems.append(f"{out} exists (rerun with --replace; the run is deterministic)")
    if problems:
        raise GlobalRefused("; ".join(problems))
    assert sha is not None
    return sha


@dataclass(slots=True)
class GlobalRun:
    split: str
    directory: Path
    fits: list[FitSample]
    gate: GateResult
    load_seconds: float
    seconds: float


def fit_manifest(s: FitSample, columns: list[str]) -> dict[str, Any]:
    f = s.fit
    return {
        "key": f.key,
        "retrain": f.retrain,
        "band": list(f.band),
        "target_form": f.target_form,
        "objective": f.objective,
        "num_leaves": f.num_leaves,
        "n_trees": f.n_trees,
        "fold_start": f.fold_start.isoformat(),
        "fold_end": f.fold_end.isoformat() if f.fold_end is not None else None,
        "eligible_pairs": s.eligible,
        "pairs": len(s.frame),
        "gate": s.gate.as_dict(),
        "gain": dict(zip(columns, s.gain.tolist(), strict=True)),
        "seconds": s.seconds,
        "file": f"{FITS_DIR}/{f.key}.parquet",
    }


def run_global(
    engine: Engine, cfg: AppConfig, region_id: str, split: str = "dev", replace: bool = False
) -> GlobalRun:
    if split != "dev":
        raise ValueError("only the development sample is computed here; the holdout is step 6's")
    ex = cfg.anomalies.explain
    sha = check_preconditions(cfg, split, replace)
    t0 = time.perf_counter()
    start, end = split_period(cfg, split)
    if end is None:
        raise ValueError(f"split {split} has no end")
    period = Period(split, start, end)
    fits = stored_fits(engine, cfg, split)
    batch_id = ex.dev_batch
    batch = read_stored_batch(cfg.data_path(cfg.paths.backtest_dir), batch_id, "demand")
    mask = batch.common_mask()
    stored_all = batch.forecasts[ex.model]
    from gridcast.features.inputs import load_context, load_feature_builder

    ctx = load_context(engine, cfg, region_id)
    builder = load_feature_builder(engine, cfg, region_id, ctx=ctx)
    columns = catalogue.feature_names(ex.variant)
    load_seconds = time.perf_counter() - t0
    log.info("explain-%s: %d fits, inputs loaded in %.0f s", split, len(fits), load_seconds)

    final = output_dir(cfg, split)
    partial = final.with_name(final.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    (partial / FITS_DIR).mkdir(parents=True)
    models = ModelCache(cfg)
    samples: list[FitSample] = []
    total = empty_gate(ex.sum_tolerance)
    try:
        for i, fit in enumerate(fits, 1):
            t1 = time.perf_counter()
            seed = cfg.anomalies.seed
            pairs = sample_pairs(fit, batch.origin_times, mask, period, ex.sample_pairs, seed)
            eligible = len(eligible_pairs(fit, batch.origin_times, mask, period))
            issue = pd.DatetimeIndex(batch.origin_times[pairs.rows])
            model = models.get(fit)
            load = time.perf_counter() - t1
            s = explain_fit(
                fit,
                model,
                builder,
                ex.variant,
                issue,
                pairs.leads,
                stored_all[pairs.rows, pairs.leads - 1],
                batch.actual[pairs.rows, pairs.leads - 1],
                ex.sum_tolerance,
                eligible,
            )
            s.seconds["load"] = load
            s.frame.to_parquet(partial / FITS_DIR / f"{fit.key}.parquet", index=False)
            total = total.merge(s.gate)
            samples.append(s)
            log.info(
                "explain-%s: fit %d/%d %s (%s, %d trees): %d pairs of %d, E1 %s, %.1f s",
                split,
                i,
                len(fits),
                fit.key,
                fit.target_form,
                fit.n_trees,
                len(s.frame),
                eligible,
                "pass" if s.gate.passed else "FAIL",
                time.perf_counter() - t1,
            )
        require(total, f"the {split} global sample")
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    seconds = time.perf_counter() - t0
    manifest = {
        "split": split,
        "region_id": region_id,
        "batch_id": batch_id,
        "model": ex.model,
        "variant": ex.variant,
        "period": [period.start.isoformat(), period.end.isoformat()],
        "sample_pairs": ex.sample_pairs,
        "seed": cfg.anomalies.seed,
        "commit": sha,
        "anomaly_config_hashes": {
            "v1": freeze_state(cfg, version=1).config_hash,
            "v2": freeze_state(cfg, version=2).config_hash,
        },
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "columns": columns,
        "gate": total.as_dict(),
        "load_seconds": load_seconds,
        "seconds": seconds,
        "fits": [fit_manifest(s, columns) for s in samples],
    }
    (partial / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    if final.exists():
        shutil.rmtree(final)
    partial.rename(final)
    return GlobalRun(split, final, samples, total, load_seconds, seconds)


def summary_lines(run: GlobalRun) -> list[str]:
    g = run.gate
    n_pairs = sum(len(s.frame) for s in run.fits)
    return [
        f"== explain-{run.split}: {len(run.fits)} fits, {n_pairs:,} pairs, "
        f"{run.seconds / 60:.1f} min (inputs {run.load_seconds:.0f} s)",
        f"   E1 {'pass' if g.passed else 'FAIL'}: forecast mismatches {g.forecast_mismatches} "
        f"of {g.n:,} (max |diff| {g.max_forecast_diff_mw:.2e} MW), SHAP-sum failures "
        f"{g.sum_failures} (max error {g.max_sum_error:.1e})",
        f"   stored under {run.directory}",
    ]


# --------------------------------------------------------------------------- reading back


@dataclass(slots=True)
class StoredSample:
    directory: Path
    manifest: dict[str, Any]

    @property
    def columns(self) -> list[str]:
        return list(self.manifest["columns"])

    @property
    def fits(self) -> list[dict[str, Any]]:
        return list(self.manifest["fits"])

    def frame(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self.directory / FITS_DIR / f"{key}.parquet")


def read_global(cfg: AppConfig, split: str = "dev") -> StoredSample:
    directory = output_dir(cfg, split)
    path = directory / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"no global SHAP sample at {directory} (run explain-{split})")
    return StoredSample(directory, json.loads(path.read_text()))
