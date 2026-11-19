"""Synthetic injection for detection power (criterion A3, m4_plan.md §2.9).

For each detector, one seeded draw of up to 1,000 **eligible** starts is shared by every
size and sign (user decision 2026-11-17). A start is eligible if every interval of its
span lies in the detector's scored period with a valid unmodified score, and the
unmodified detector raised no notable alert (p ≤ 0.001, scored period or not) within the
span ± 1 h. D2's starts are the first target of an issue time, so its six targets are
exactly the leads 1–6 of one stored forecast.

Each injection is scored alone. The profiles, the D5 baseline, the stored bands, the
Isolation Forest models and the p-value windows are the unmodified ones; an injection only
changes data from its start onwards, so this is what a live system would have computed for
the span. An injection is **detected** if the detector raises at least one notable alert
inside its span.

| Detector | Injection |
|---|---|
| D1, D3, D6 | add a signed MW size to demand (1 h, 2 h, 1 h) |
| D2 | add a signed MW size to demand for the six targets of one issue time (30 min) |
| D4 | set the price to a fixed $/MWh value for 30 min |
| D5 | alternate the price between its actual value + A and − A for 1 h |
| D5, reported only | hold the price at its value at the start for 3 h (a sudden calm) |
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridcast.anomalies.detectors import (
    DEMAND_LEVEL,
    PRICE_LEVEL,
    PRICE_VOLATILITY,
    RESIDUAL,
    RESIDUAL_DAYAHEAD,
    RESIDUAL_SHORT,
    Recipe,
    price_x,
    residual_score,
    v1_recipe,
)
from gridcast.anomalies.pvalues import Queries, window_pvalues
from gridcast.anomalies.run import DetectorRun, Split, window_intervals
from gridcast.anomalies.series import MarketSeries, epoch_minutes
from gridcast.config import AnomaliesConfig

DEMAND_ADD = "demand_add"
PRICE_SET = "price_set"
PRICE_ALTERNATE = "price_alternate"
PRICE_FLAT = "price_flat"
FLAT_LABEL = "price_volatility_flat"


@dataclass(frozen=True, slots=True)
class InjectionSpec:
    label: str  # the detector's name, or FLAT_LABEL
    detector: str  # the detector judged
    kind: str
    span: int  # intervals
    sizes: tuple[float, ...]
    aligned: bool  # starts at the first target of an issue time (D2)
    seed_index: int  # the draw's seed is (anomalies.seed, seed_index)
    reported_only: bool = False


def injection_specs(cfg: AnomaliesConfig, recipe: Recipe | None = None) -> list[InjectionSpec]:
    """§2.9's injections. D6's sizes are ``injection.isolation_forest`` in every version (the
    config names D6's slot); its label is the recipe's multivariate detector."""
    inj = cfg.injection
    recipe = v1_recipe(cfg) if recipe is None else recipe

    def spec(
        detector: str, kind: str, index: int, aligned: bool = False, setting: str | None = None
    ) -> InjectionSpec:
        s = getattr(inj, setting or detector)
        return InjectionSpec(
            detector, detector, kind, s.minutes // 5, tuple(s.sizes), aligned, index
        )

    return [
        spec(DEMAND_LEVEL, DEMAND_ADD, 1),
        spec(RESIDUAL_SHORT, DEMAND_ADD, 2, aligned=True),
        spec(RESIDUAL_DAYAHEAD, DEMAND_ADD, 3),
        spec(PRICE_LEVEL, PRICE_SET, 4),
        spec(PRICE_VOLATILITY, PRICE_ALTERNATE, 5),
        spec(recipe.multivariate, DEMAND_ADD, 6, setting="isolation_forest"),
        InjectionSpec(
            FLAT_LABEL,
            PRICE_VOLATILITY,
            PRICE_FLAT,
            inj.flat_price_minutes // 5,
            (0.0,),
            False,
            7,
            reported_only=True,
        ),
    ]


# ----------------------------------------------------------------------------- placement


def _all_in_window(flags: np.ndarray, before: int, length: int, after: int) -> np.ndarray:
    """For each start i in [0, n − length]: any flag in [i − before, i + length − 1 + after]."""
    padded = np.concatenate([np.zeros(before, bool), flags, np.zeros(after, bool)])
    view = np.lib.stride_tricks.sliding_window_view(padded, before + length + after)
    return view.any(axis=1)[: len(flags) - length + 1]


def eligible_starts(
    spec: InjectionSpec,
    run: DetectorRun,
    scored: np.ndarray,
    notable: float,
    guard: int,
) -> np.ndarray:
    """Interval indices where an injection of ``spec`` may start (§2.9)."""
    n = len(run.ends)
    if n < spec.span:
        return np.zeros(0, dtype=np.int64)
    usable = scored & np.isfinite(run.scores[spec.detector])
    span_ok = ~_all_in_window(~usable, 0, spec.span, 0)
    with np.errstate(invalid="ignore"):
        alert = np.asarray(run.pvalues[spec.detector] <= notable)
    quiet = ~_all_in_window(alert, guard, spec.span, guard)
    ok = span_ok & quiet
    if spec.aligned:
        minutes = epoch_minutes(run.ends[: n - spec.span + 1])
        ok &= (minutes - 5) % 30 == 0
    return np.flatnonzero(ok)


def draw_starts(eligible: np.ndarray, n_starts: int, seed: int, index: int) -> np.ndarray:
    rng = np.random.default_rng([seed, index])
    k = min(n_starts, len(eligible))
    return np.sort(rng.choice(eligible, size=k, replace=False)) if k else eligible[:0]


# ----------------------------------------------------------------------------- re-scoring


def _volatility_rows(price: np.ndarray, cfg: AnomaliesConfig, scale_aud: float) -> np.ndarray:
    """v over the last ``window`` columns-worth of changes: price is ``(k, W + span)``."""
    w = cfg.price_volatility.window_intervals
    x = price_x(price, scale_aud)
    dx = np.diff(x, axis=1)  # (k, W + span − 1)
    sq = np.lib.stride_tricks.sliding_window_view(dx * dx, w, axis=1)  # (k, span, W)
    return np.sqrt(sq.mean(axis=2))


def injected_scores(
    spec: InjectionSpec,
    size: float,
    starts: np.ndarray,
    series: MarketSeries,
    run: DetectorRun,
    cfg: AnomaliesConfig,
) -> np.ndarray:
    """``(k, span)`` scores of the detector over each injected span."""
    st = run.state
    rows = starts[:, None] + np.arange(spec.span)[None, :]
    k_mad = cfg.profile.mad_consistency
    if spec.kind == DEMAND_ADD:
        y = series.demand[rows] + size
        if spec.detector == DEMAND_LEVEL:
            prof = st.demand_profile
            scale = np.maximum(k_mad * prof.mad[rows], cfg.demand_level.floor_mw)
            return np.abs((y - prof.median[rows]) / scale)
        if spec.detector in RESIDUAL:
            lo, hi, _ = st.bands[spec.detector]
            return residual_score(y, lo[rows], hi[rows], cfg.residual.width_floor_mw)
        if spec.detector == run.recipe.multivariate:
            return _multivariate_scores(rows, y, series, run, cfg)
    if spec.kind == PRICE_SET:
        x = price_x(np.full(rows.shape, size), run.recipe.price_level_scale_aud)
        prof = st.price_profile
        scale = np.maximum(k_mad * prof.mad[rows], cfg.price_level.floor)
        return np.abs((x - prof.median[rows]) / scale)
    if spec.kind in (PRICE_ALTERNATE, PRICE_FLAT):
        w = cfg.price_volatility.window_intervals
        ctx = starts[:, None] + np.arange(-w, spec.span)[None, :]
        price = series.price[ctx].astype(np.float64)
        if spec.kind == PRICE_ALTERNATE:
            signs = np.where(np.arange(spec.span) % 2 == 0, 1.0, -1.0)
            price[:, w:] += size * signs[None, :]
        else:
            price[:, w + 1 :] = price[:, [w]]
        v = _volatility_rows(price, cfg, run.recipe.price_volatility_scale_aud)
        eps = cfg.price_volatility.epsilon
        return np.abs(np.log((v + eps) / (st.b[rows] + eps)))
    raise ValueError(f"no injection rule for {spec.kind} on {spec.detector}")


def _multivariate_scores(
    rows: np.ndarray, y: np.ndarray, series: MarketSeries, run: DetectorRun, cfg: AnomaliesConfig
) -> np.ndarray:
    """D6 features recomputed from the injected demand, scored by each row's month model."""
    st = run.state
    k_mad = cfg.profile.mad_consistency
    iso = cfg.isolation_forest
    f = st.features[rows].copy()  # (k, span, 9)
    prof = st.demand_profile
    f[..., 0] = (y - prof.median[rows]) / np.maximum(
        k_mad * prof.mad[rows], cfg.demand_level.floor_mw
    )
    before = series.demand[rows[:, 0] - 1]
    prev = np.concatenate([before[:, None], y[:, :-1]], axis=1)
    f[..., 3] = (y - prev) / st.change_scale[rows]
    margin = series.available[rows] - y
    mp = st.margin_profile
    f[..., 6] = (margin - mp.median[rows]) / np.maximum(k_mad * mp.mad[rows], iso.z_floor_mw)
    out = np.full(rows.shape, np.nan)
    scorers = st.scorers()
    months = st.lay.nem_month[rows]
    for month in np.unique(months):
        sel = months == month
        out[sel] = scorers[int(month)].score(f[sel])
    return out


def _month_pvalues(rows: np.ndarray, scores: np.ndarray, run: DetectorRun) -> np.ndarray:
    """p-values against each row's month model's training scores (v1's Isolation Forest)."""
    out = np.full(rows.shape, np.nan)
    months = run.state.lay.nem_month[rows]
    for month in np.unique(months):
        sel = months == month
        out[sel] = run.state.models[int(month)].pvalues(scores[sel])
    return out


# ----------------------------------------------------------------------------- the run


@dataclass(slots=True)
class _Batch:
    spec: InjectionSpec
    size: float
    starts: np.ndarray
    scores: np.ndarray  # (k, span)
    query_slice: slice | None  # into the extra queries (rolling detectors)


def run_injections(
    series: MarketSeries,
    run: DetectorRun,
    split: Split,
    cfg: AnomaliesConfig,
    n_starts: int | None = None,
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One row per injection: label, detector, size, start, detected, min_p, n_eligible."""
    n_starts = cfg.injection.n_starts if n_starts is None else n_starts
    guard = cfg.injection.guard_minutes // 5
    notable = cfg.pvalues.notable
    masks = split.masks(run.ends)
    batches: list[_Batch] = []
    q_rows, q_cols, q_vals = [], [], []
    offset = 0
    eligible_count: dict[str, int] = {}
    rolling = run.recipe.rolling
    for spec in injection_specs(cfg, run.recipe):
        if labels is not None and spec.label not in labels:
            continue
        eligible = eligible_starts(spec, run, masks[spec.detector], notable, guard)
        eligible_count[spec.label] = len(eligible)
        starts = draw_starts(eligible, n_starts, cfg.seed, spec.seed_index)
        for size in spec.sizes:
            scores = injected_scores(spec, size, starts, series, run, cfg)
            sl = None
            if spec.detector in rolling and scores.size:
                rows = starts[:, None] + np.arange(spec.span)[None, :]
                q_rows.append(rows.ravel())
                q_cols.append(np.full(rows.size, rolling.index(spec.detector)))
                q_vals.append(scores.ravel())
                sl = slice(offset, offset + rows.size)
                offset += rows.size
            batches.append(_Batch(spec, size, starts, scores, sl))

    extra_p = np.zeros(0)
    if offset:
        matrix = np.column_stack([run.scores[d] for d in rolling])
        queries = Queries(np.concatenate(q_rows), np.concatenate(q_cols), np.concatenate(q_vals))
        extra_p = window_pvalues(
            matrix, np.isfinite(matrix), window_intervals(cfg), queries
        ).extra_p

    parts = []
    for b in batches:
        if b.starts.size == 0:
            continue
        if b.query_slice is not None:
            p = extra_p[b.query_slice].reshape(b.scores.shape)
        else:
            rows = b.starts[:, None] + np.arange(b.spec.span)[None, :]
            p = _month_pvalues(rows, b.scores, run)
        min_p = np.nanmin(p, axis=1)
        parts.append(
            pd.DataFrame(
                {
                    "label": b.spec.label,
                    "detector": b.spec.detector,
                    "size": b.size,
                    "start": run.ends[b.starts],
                    "detected": min_p <= notable,
                    "min_p": min_p,
                    "n_eligible": eligible_count[b.spec.label],
                    "reported_only": b.spec.reported_only,
                }
            )
        )
    columns = [
        "label",
        "detector",
        "size",
        "start",
        "detected",
        "min_p",
        "n_eligible",
        "reported_only",
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def power_table(results: pd.DataFrame) -> pd.DataFrame:
    """Detected share per (label, size), with the number of injections."""
    if results.empty:
        return pd.DataFrame(columns=["label", "size", "n", "power"])
    g = results.groupby(["label", "size"], sort=False)["detected"]
    return g.agg(n="size", power="mean").reset_index()
