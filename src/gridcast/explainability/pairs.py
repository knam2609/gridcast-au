"""Explain a set of stored (issue time, lead) pairs with the fits that forecast them.

Each pair goes to the one fit whose fold holds its issue time and whose band holds its lead
(:func:`~gridcast.explainability.stored.locate`). Its feature row is rebuilt, TreeSHAP runs
on the fit's stored booster, and E1's gate compares the recomputed forecast with the stored
one (:mod:`gridcast.explainability.treeshap`). The gate result is returned per fit and in
total; :func:`~gridcast.explainability.treeshap.require` withholds explanations on failure.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gridcast.config import AppConfig
from gridcast.explainability.sample import feature_rows, origin_positions
from gridcast.explainability.stored import StoredFit, load_model, locate
from gridcast.explainability.treeshap import (
    Explanation,
    GateResult,
    empty_gate,
    explain,
    gate,
)
from gridcast.features import catalogue
from gridcast.features.builder import FeatureBuilder
from gridcast.models.ml import LightGBMModel


@dataclass(slots=True)
class PairExplanations:
    issue_times: pd.DatetimeIndex
    leads: np.ndarray
    fit_index: np.ndarray  # position in ``fits`` per pair
    fits: list[StoredFit]
    explanation: Explanation | None  # rows in the pairs' order (None: no pairs)
    gate: GateResult
    gate_by_fit: dict[str, GateResult]
    seconds: dict[str, dict[str, float]] = field(default_factory=dict)  # per fit key


class ModelCache:
    """Loads each stored booster once."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.columns = catalogue.feature_names(cfg.anomalies.explain.variant)
        self._models: dict[str, LightGBMModel] = {}

    def get(self, fit: StoredFit) -> LightGBMModel:
        if fit.key not in self._models:
            self._models[fit.key] = load_model(
                fit, self.columns, self.cfg.ml.lightgbm, self.cfg.ml.seed
            )
        return self._models[fit.key]


def _concat(parts: list[tuple[np.ndarray, Explanation]], n: int) -> Explanation:
    """One explanation whose row ``i`` is pair ``i``."""
    first = parts[0][1]
    p = len(first.columns)
    forecast, anchor, raw, base = (np.full(n, np.nan) for _ in range(4))
    contributions = np.full((n, p), np.nan)
    forms = set()
    for idx, ex in parts:
        forecast[idx], anchor[idx], raw[idx], base[idx] = ex.forecast, ex.anchor, ex.raw, ex.base
        contributions[idx] = ex.contributions
        forms.add(ex.target_form)
    form = forms.pop() if len(forms) == 1 else "mixed"
    return Explanation(first.columns, forecast, anchor, raw, base, contributions, form)


def explain_pairs(
    builder: FeatureBuilder,
    fits: Sequence[StoredFit],
    issue_times: pd.DatetimeIndex,
    leads: np.ndarray,
    stored: np.ndarray,
    cfg: AppConfig,
    models: ModelCache | None = None,
) -> PairExplanations:
    """Explain every pair; ``stored`` holds M2's stored forecast of each pair."""
    ex_cfg = cfg.anomalies.explain
    leads = np.asarray(leads, dtype=np.int64)
    stored = np.asarray(stored)
    if not (len(issue_times) == len(leads) == len(stored)):
        raise ValueError("one lead and one stored forecast per issue time")
    models = models if models is not None else ModelCache(cfg)
    fits = list(fits)
    where = locate(fits, issue_times, leads)
    if np.any(where < 0):
        first = int(np.flatnonzero(where < 0)[0])
        raise LookupError(
            f"{int((where < 0).sum())} pairs have no stored fit, e.g. issue time "
            f"{issue_times[first]} lead {leads[first]}"
        )
    origins = origin_positions(builder, issue_times) if len(leads) else np.array([], np.int64)
    parts: list[tuple[np.ndarray, Explanation]] = []
    total = empty_gate(ex_cfg.sum_tolerance)
    by_fit: dict[str, GateResult] = {}
    seconds: dict[str, dict[str, float]] = {}
    for k in np.unique(where):
        fit = fits[int(k)]
        idx = np.flatnonzero(where == k)
        t0 = time.perf_counter()
        model = models.get(fit)
        t1 = time.perf_counter()
        X = feature_rows(builder, ex_cfg.variant, origins[idx], leads[idx])
        t2 = time.perf_counter()
        if not np.isfinite(X).all():
            raise ValueError(
                f"fit {fit.key}: {int((~np.isfinite(X).all(axis=1)).sum())} explained pairs "
                "have a missing feature, so M2 stored no forecast for them"
            )
        ex = explain(model, X)
        t3 = time.perf_counter()
        result = gate(ex, stored[idx], ex_cfg.sum_tolerance)
        by_fit[fit.key] = result
        total = total.merge(result)
        parts.append((idx, ex))
        seconds[fit.key] = {"load": t1 - t0, "build": t2 - t1, "shap": t3 - t2}
    return PairExplanations(
        issue_times=issue_times,
        leads=leads,
        fit_index=where,
        fits=fits,
        explanation=_concat(parts, len(leads)) if parts else None,
        gate=total,
        gate_by_fit=by_fit,
        seconds=seconds,
    )
