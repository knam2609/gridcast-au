"""Detector scores D1–D6 (m4_plan.md §2.3). A larger score means more unusual.

Every score at interval ``t`` uses only data known at ``t``'s end:

* **D1** ``demand_level``: |z| of demand against its profile split by day type.
* **D2** ``demand_residual_short`` / **D3** ``demand_residual_dayahead``: the distance of
  demand outside M3's raw quantile LightGBM C 2.5–97.5% band, over the band's width, using
  the one issue time whose lead to ``t`` lies in the detector's six leads (issue times are
  30 minutes apart). Only M2's scored pairs (common mask) are valid.
* **D4** ``price_level``: |z| of x = asinh(price / $10) against its profile (all day types).
* **D5** ``price_volatility``: |ln((v + ε) / (b + ε))|, v = RMS of x's 5-minute changes over
  the last hour, b = median of v over the 7 days ending 1 h before ``t``.
* **D6** ``isolation_forest``: −``score_samples`` of the month's Isolation Forest, refitted
  at each NEM-month start on the previous 26 weeks (:mod:`gridcast.anomalies.isolation`).

:class:`DetectorState` keeps the unmodified profiles, baselines, bands and models, so the
injection harness can re-score modified data against them (§2.9).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridcast.anomalies.isolation import MonthModel, fit_month_models, score_months
from gridcast.anomalies.profiles import Profile, daily_robust_scale, slot_profile
from gridcast.anomalies.series import (
    INTERVALS_PER_DAY,
    Layout,
    MarketSeries,
    QuantileBand,
    epoch_minutes,
    layout,
)
from gridcast.config import AnomaliesConfig

DEMAND_LEVEL = "demand_level"
RESIDUAL_SHORT = "demand_residual_short"
RESIDUAL_DAYAHEAD = "demand_residual_dayahead"
PRICE_LEVEL = "price_level"
PRICE_VOLATILITY = "price_volatility"
ISOLATION_FOREST = "isolation_forest"
LINE_NOTABLE = "price_line_300"
LINE_SEVERE = "price_line_5000"

# Calibrated detectors in §2.3's order; the first five use rolling windows.
DETECTORS = (
    DEMAND_LEVEL,
    RESIDUAL_SHORT,
    RESIDUAL_DAYAHEAD,
    PRICE_LEVEL,
    PRICE_VOLATILITY,
    ISOLATION_FOREST,
)
ROLLING = DETECTORS[:5]
LINES = (LINE_NOTABLE, LINE_SEVERE)
RESIDUAL = (RESIDUAL_SHORT, RESIDUAL_DAYAHEAD)
MARKET = (DEMAND_LEVEL, PRICE_LEVEL, PRICE_VOLATILITY, ISOLATION_FOREST)

# §2.4's category labels, in its order.
CATEGORIES = (
    "unusual demand",
    "unexpected demand (short-term)",
    "unexpected demand (day-ahead)",
    "high price",
    "unusual price",
    "volatility change",
    "multivariate",
)
CATEGORY = {
    DEMAND_LEVEL: CATEGORIES[0],
    RESIDUAL_SHORT: CATEGORIES[1],
    RESIDUAL_DAYAHEAD: CATEGORIES[2],
    LINE_NOTABLE: CATEGORIES[3],
    LINE_SEVERE: CATEGORIES[3],
    PRICE_LEVEL: CATEGORIES[4],
    PRICE_VOLATILITY: CATEGORIES[5],
    ISOLATION_FOREST: CATEGORIES[6],
}

D6_FEATURES = (
    "demand_z",
    "price_z",
    "volatility_log_ratio",
    "demand_change_scaled",
    "net_interchange_z",
    "dispatchable_z",
    "margin_z",
    "wind_curtailment",
    "solar_curtailment",
)


# ----------------------------------------------------------------------------- pieces


def price_x(price: np.ndarray, scale: float) -> np.ndarray:
    """asinh(price / scale): about linear near 0, logarithmic for spikes, signed."""
    return np.arcsinh(np.asarray(price, dtype=np.float64) / scale)


def first_difference(values: np.ndarray) -> np.ndarray:
    """``values[t] − values[t − 1]``; NaN at the first interval."""
    values = np.asarray(values, dtype=np.float64)
    out = np.full(len(values), np.nan)
    out[1:] = values[1:] - values[:-1]
    return out


def volatility(x: np.ndarray, window: int) -> np.ndarray:
    """RMS of x's 5-minute changes over the ``window`` intervals ending at ``t``.

    Computed directly from each window (no running sums), so a value depends only on its
    own window, bit for bit.
    """
    dx = first_difference(x)
    out = np.full(len(dx), np.nan)
    if len(dx) >= window:
        sq = np.lib.stride_tricks.sliding_window_view(dx * dx, window)
        out[window - 1 :] = np.sqrt(sq.mean(axis=1))
    return out


def volatility_baseline(v: np.ndarray, n: int, lag: int) -> np.ndarray:
    """Median of v over the ``n`` intervals ending ``lag`` intervals before ``t``.

    Missing until a full window of finite values exists.
    """
    med = pd.Series(v).rolling(n, min_periods=n).median().to_numpy()
    out = np.full(len(v), np.nan)
    if lag < len(v):
        out[lag:] = med[: len(v) - lag]
    return out


def log_ratio(v: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    return np.log((v + eps) / (b + eps))


def curtailment(cleared: np.ndarray, uigf: np.ndarray, min_uigf: float) -> np.ndarray:
    """1 − cleared / UIGF where UIGF ≥ ``min_uigf``, else 0 (§2.3, against UIGF: step 1b)."""
    cleared = np.asarray(cleared, dtype=np.float64)
    uigf = np.asarray(uigf, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(uigf >= min_uigf, 1.0 - cleared / uigf, 0.0)
    return np.where(np.isfinite(cleared) & np.isfinite(uigf), share, np.nan)


def band_at_targets(
    band: QuantileBand, ends: pd.DatetimeIndex, leads: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The band's lower and upper bound for each target interval, and whether it is valid.

    Target ``t`` uses the one issue time ``o`` with ``t − o`` in ``leads`` (× 5 min). Issue
    times are 30 minutes apart, so ``o = floor₃₀(t − 5 min × leads[0])``.
    """
    n = len(ends)
    lower, upper = np.full(n, np.nan), np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)
    if len(band.issue_times) == 0:
        return lower, upper, ok
    minutes = epoch_minutes(ends)
    shifted = minutes - 5 * leads[0]
    issue = shifted - shifted % 30
    lead = (minutes - issue) // 5  # in [leads[0], leads[0] + 5]
    issue_minutes = epoch_minutes(band.issue_times)
    row = np.searchsorted(issue_minutes, issue)
    found = (row < len(issue_minutes)) & (lead <= band.n_leads)
    found[found] &= issue_minutes[row[found]] == issue[found]
    r, h = row[found], lead[found] - 1
    lower[found] = band.lower[r, h]
    upper[found] = band.upper[r, h]
    ok[found] = band.valid[r, h]
    return lower, upper, ok


def residual_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, floor: float) -> np.ndarray:
    """max(lower − y, y − upper) / max(upper − lower, floor): > 0 outside the band."""
    return np.maximum(lower - y, y - upper) / np.maximum(upper - lower, floor)


# ----------------------------------------------------------------------------- state


@dataclass(slots=True)
class DetectorState:
    """Everything the scores were computed from that an injection must leave unmodified."""

    lay: Layout
    demand_profile: Profile  # D1 (by day type)
    x: np.ndarray  # asinh(price / scale)
    price_profile: Profile  # D4 (all day types)
    v: np.ndarray  # D5 current volatility
    b: np.ndarray  # D5 baseline
    bands: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]  # D2/D3: lower, upper, ok
    change_scale: np.ndarray  # D6: robust scale of the 5-minute demand changes
    margin_profile: Profile  # D6: available generation minus demand
    features: np.ndarray  # (n, 9) D6 features
    models: dict[int, MonthModel]  # D6 by NEM month


def demand_z(demand: np.ndarray, state: DetectorState, cfg: AnomaliesConfig) -> np.ndarray:
    return state.demand_profile.z(demand, cfg.profile.mad_consistency, cfg.demand_level.floor_mw)


def price_z(x: np.ndarray, state: DetectorState, cfg: AnomaliesConfig) -> np.ndarray:
    return state.price_profile.z(x, cfg.profile.mad_consistency, cfg.price_level.floor)


def margin_z(margin: np.ndarray, state: DetectorState, cfg: AnomaliesConfig) -> np.ndarray:
    floor = cfg.isolation_forest.z_floor_mw
    return state.margin_profile.z(margin, cfg.profile.mad_consistency, floor)


def _band_scores(
    series: MarketSeries, band: QuantileBand | None, cfg: AnomaliesConfig
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    """D2 and D3: the band at each target, and the scores where M2's mask allows."""
    n = len(series)
    bands: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    residual: dict[str, np.ndarray] = {}
    for name, leads in (
        (RESIDUAL_SHORT, cfg.residual.short_leads),
        (RESIDUAL_DAYAHEAD, cfg.residual.dayahead_leads),
    ):
        lo, hi, ok = (
            band_at_targets(band, series.ends, leads)
            if band is not None
            else (np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=bool))
        )
        ok = ok & np.isfinite(series.demand)
        bands[name] = (lo, hi, ok)
        s = residual_score(series.demand, lo, hi, cfg.residual.width_floor_mw)
        residual[name] = np.where(ok, s, np.nan)
    return bands, residual


def compute_scores(
    series: MarketSeries,
    band: QuantileBand | None,
    cfg: AnomaliesConfig,
    market: bool = True,
) -> tuple[dict[str, np.ndarray], DetectorState]:
    """Every calibrated detector's score per interval (NaN where not valid).

    ``market=False`` computes only D2 and D3: the market detectors' scores, profiles and
    models are never computed (their scores are NaN). The warm-up smoke run uses this to
    score D2/D3 over data that lies in the market detectors' development period.
    """
    lay = layout(series, cfg.profile.slot_minutes)
    n = len(series)
    days, k = cfg.profile.days, cfg.profile.mad_consistency
    bands, residual = _band_scores(series, band, cfg)
    nan = np.full(n, np.nan)
    if not market:
        empty = Profile(nan, nan)
        state = DetectorState(
            lay=lay,
            demand_profile=empty,
            x=nan,
            price_profile=empty,
            v=nan,
            b=nan,
            bands=bands,
            change_scale=nan,
            margin_profile=empty,
            features=np.full((n, len(D6_FEATURES)), np.nan),
            models={},
        )
        scores = {d: nan for d in DETECTORS} | residual
        return scores, state

    # D1
    demand_profile = slot_profile(series.demand, lay, days, by_day_type=True)
    z1 = demand_profile.z(series.demand, k, cfg.demand_level.floor_mw)
    # D4
    x = price_x(series.price, cfg.price_level.scale_aud)
    price_profile = slot_profile(x, lay, days, by_day_type=False)
    z4 = price_profile.z(x, k, cfg.price_level.floor)
    # D5
    vol = cfg.price_volatility
    v = volatility(x, vol.window_intervals)
    b = volatility_baseline(v, vol.baseline_days * INTERVALS_PER_DAY, vol.baseline_lag_intervals)
    r5 = log_ratio(v, b, vol.epsilon)
    # D6
    iso = cfg.isolation_forest
    change = first_difference(series.demand)
    change_scale = daily_robust_scale(change, lay, days, k, iso.change_floor_mw)
    margin = series.available - series.demand
    margin_profile = slot_profile(margin, lay, days, by_day_type=False)
    features = np.column_stack(
        [
            z1,
            z4,
            r5,
            change / change_scale,
            slot_profile(series.net_interchange, lay, days, False).z(
                series.net_interchange, k, iso.z_floor_mw
            ),
            slot_profile(series.dispatchable, lay, days, False).z(
                series.dispatchable, k, iso.z_floor_mw
            ),
            margin_profile.z(margin, k, iso.z_floor_mw),
            curtailment(series.wind_cleared, series.wind_uigf, iso.curtailment_min_uigf_mw),
            curtailment(series.solar_cleared, series.solar_uigf, iso.curtailment_min_uigf_mw),
        ]
    )
    models = fit_month_models(features, series.ends, lay, iso, cfg.seed)
    s6 = score_months(features, lay, models)

    state = DetectorState(
        lay=lay,
        demand_profile=demand_profile,
        x=x,
        price_profile=price_profile,
        v=v,
        b=b,
        bands=bands,
        change_scale=change_scale,
        margin_profile=margin_profile,
        features=features,
        models=models,
    )
    scores = {
        DEMAND_LEVEL: np.abs(z1),
        RESIDUAL_SHORT: residual[RESIDUAL_SHORT],
        RESIDUAL_DAYAHEAD: residual[RESIDUAL_DAYAHEAD],
        PRICE_LEVEL: np.abs(z4),
        PRICE_VOLATILITY: np.abs(r5),
        ISOLATION_FOREST: s6,
    }
    return scores, state
