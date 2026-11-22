"""SHAP smoke run on one development retrain (m4_plan.md step 3): bugs and cost only.

User decision 2026-11-19 (§1, "Clarified in step 3"): the run explains the global sample of
retrain 2022-11-01, whose issue times fall in D2/D3's warm-up, and prints **only** E1's
results and the timings, never a SHAP summary. It writes nothing.

The cost projection scales each fit's measured seconds per (row × tree) to every stored fit
of both batches at the full sample size; the rate is taken from a smoke fit with the same
leaf count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from gridcast.config import AppConfig
from gridcast.explainability.pairs import ModelCache, PairExplanations, explain_pairs
from gridcast.explainability.sample import Period, sample_pairs
from gridcast.explainability.stored import StoredFit, stored_fits
from gridcast.features.inputs import load_context, load_feature_builder
from gridcast.forecasting.origins import nem_midnight_utc
from gridcast.intervals.inputs import read_stored_batch


@dataclass(slots=True)
class SmokeFit:
    fit: StoredFit
    eligible_issue_times: int
    pairs: int
    result: PairExplanations


@dataclass(slots=True)
class ExplainSmoke:
    retrain: str
    fits: list[SmokeFit]
    load_seconds: float
    projection: dict[str, float]  # split -> projected seconds for the full global sample
    n_fits: dict[str, int]


def dev_period(cfg: AppConfig) -> Period:
    return Period(
        "dev",
        nem_midnight_utc(str(cfg.ml.first_issue)),
        nem_midnight_utc(str(cfg.backtest.holdout_start)),
    )


def _rate(sf: SmokeFit) -> float:
    """Seconds per (row × tree) of the predict + SHAP step."""
    secs = sf.result.seconds[sf.fit.key]["shap"]
    return secs / max(sf.pairs * sf.fit.n_trees, 1)


def run_explain_smoke(
    engine: Engine, cfg: AppConfig, region_id: str, retrain: str = "2022-11-01"
) -> ExplainSmoke:
    ex = cfg.anomalies.explain
    t0 = time.perf_counter()
    dev_fits = stored_fits(engine, cfg, "dev")
    holdout_fits = stored_fits(engine, cfg, "holdout")
    chosen = [f for f in dev_fits if f.retrain == retrain]
    if not chosen:
        raise LookupError(f"run 2 has no retrain {retrain}")
    period = dev_period(cfg)
    fold_end = chosen[0].fold_end
    if fold_end is None or fold_end > nem_midnight_utc(str(cfg.anomalies.residual_first_day)):
        raise ValueError(
            f"retrain {retrain}'s issue times reach past D2/D3's warm-up "
            f"(ends {cfg.anomalies.residual_first_day}); the smoke run stays inside it"
        )
    batch = read_stored_batch(cfg.data_path(cfg.paths.backtest_dir), ex.dev_batch, "demand")
    batch = batch.restrict(chosen[0].fold_start, fold_end)
    mask = batch.common_mask()
    stored = batch.forecasts[ex.model]
    ctx = load_context(engine, cfg, region_id)
    builder = load_feature_builder(engine, cfg, region_id, ctx=ctx)
    load_seconds = time.perf_counter() - t0

    models = ModelCache(cfg)
    out: list[SmokeFit] = []
    for fit in chosen:
        pairs = sample_pairs(
            fit, batch.origin_times, mask, period, ex.sample_pairs, cfg.anomalies.seed
        )
        issue = pd.DatetimeIndex(batch.origin_times[pairs.rows])
        res = explain_pairs(
            builder,
            [fit],
            issue,
            pairs.leads,
            stored[pairs.rows, pairs.leads - 1],
            cfg,
            models,
        )
        out.append(SmokeFit(fit, int(np.unique(pairs.rows).size), len(pairs), res))

    rates = {sf.fit.num_leaves: _rate(sf) for sf in out}
    fallback = max(rates.values())
    projection = {}
    for split, fits in (("dev", dev_fits), ("holdout", holdout_fits)):
        projection[split] = float(
            sum(rates.get(f.num_leaves, fallback) * ex.sample_pairs * f.n_trees for f in fits)
        )
    return ExplainSmoke(
        retrain,
        out,
        load_seconds,
        projection,
        {"dev": len(dev_fits), "holdout": len(holdout_fits)},
    )


def summarise(smoke: ExplainSmoke) -> list[str]:
    """Console lines: E1 and timings only (no SHAP values, by the user's decision)."""
    lines = [
        f"== SHAP smoke run: retrain {smoke.retrain} (run 2), inputs loaded in "
        f"{smoke.load_seconds:.0f} s",
    ]
    total_pass = True
    for sf in smoke.fits:
        f, g = sf.fit, sf.result.gate
        s = sf.result.seconds[f.key]
        total_pass &= g.passed
        lines.append(
            f"   {f.key}: {f.target_form:8s} {f.objective} leaves {f.num_leaves:3d} "
            f"trees {f.n_trees:4d} | pairs {sf.pairs:,} over {sf.eligible_issue_times:,} issue "
            f"times | E1 {'pass' if g.passed else 'FAIL'}: forecast mismatches "
            f"{g.forecast_mismatches}, max |diff| {g.max_forecast_diff_mw:.2e} MW, sum failures "
            f"{g.sum_failures}, max sum error {g.max_sum_error:.1e} | load {s['load']:.1f} s, "
            f"build {s['build']:.1f} s, predict+SHAP {s['shap']:.1f} s"
        )
    lines.append(f"   E1 on every smoke pair: {'pass' if total_pass else 'FAIL'}")
    for split, secs in smoke.projection.items():
        lines.append(
            f"   projected predict+SHAP for the full {split} global sample "
            f"({smoke.n_fits[split]} fits): {secs / 60:.1f} min (feature building extra)"
        )
    return lines
