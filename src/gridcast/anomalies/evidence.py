"""Event evidence and facts-only summaries (m4_plan.md §2.5; criterion A4).

Each event's evidence is a set of **fields**. A field is either a fact (a value, its unit,
its source, and whether it was *known at the time* or is *retrospective*, i.e. published
later) or an explicit ``{"unavailable": reason}``. :data:`REQUIRED_FIELDS` lists every field
of §2.5. :func:`missing_fields` is A4's completeness check, and :func:`banned_in` is its
phrase test.

Decisions of the step-3 question round (§1, "Clarified in step 3"):

* a demand event's SHAP breakdown, and the forecasts beside it, are taken at its **demand
  peak**: the D1–D3 alert with the lowest p, with §2.4's tie rule;
* summaries have **three sentences**: the lead fact for the peak's category; the other
  market variable over the same 5 minutes; the operator flags in the event's span.

Everything here is pure: :class:`EvidenceContext` holds the inputs, already looked up for
the pairs and intervals the events need (:mod:`gridcast.anomalies.evidence_inputs` loads
them). Times are interval ends. Local time is the series' zone (Australia/Melbourne); NEM
time is UTC+10.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from gridcast.anomalies.detectors import (
    CATEGORY,
    DEMAND_LEVEL,
    ISOLATION_FOREST,
    LINE_NOTABLE,
    LINE_SEVERE,
    LINES,
    PRICE_LEVEL,
    PRICE_VOLATILITY,
    RESIDUAL_DAYAHEAD,
    RESIDUAL_SHORT,
    price_x,
    volatility,
    volatility_baseline,
)
from gridcast.anomalies.events import ORDER, Event
from gridcast.anomalies.profiles import SlotValues
from gridcast.anomalies.pvalues import NOTABLE, SEVERE, TIER_NAMES
from gridcast.anomalies.run import Detection
from gridcast.anomalies.series import INTERVALS_PER_DAY, Layout, MarketSeries, layout
from gridcast.config import AnomaliesConfig
from gridcast.evaluation.report import money
from gridcast.timeutils import INTERVAL, NEM_TZ

KNOWN = "known at the time"
RETRO = "retrospective"
VERSION = 1
DEMAND_DETECTORS = (DEMAND_LEVEL, RESIDUAL_SHORT, RESIDUAL_DAYAHEAD)
WEEK_INTERVALS = 7 * INTERVALS_PER_DAY

# APCFLAG bits, short names for the summaries (full meanings: anomalies.reference.APC_BITS).
APC_SHORT = {
    1: "APC/APF binding",
    2: "VoLL override",
    4: "MPC/MPF binding",
    8: "price manually overwritten",
    16: "price scaled by the inter-regional loss factor",
}

SOURCES = {
    "demand": "market_intervals.total_demand_mw (DISPATCHREGIONSUM.TOTALDEMAND)",
    "price": "market_intervals.rrp_aud_mwh (DISPATCHPRICE.RRP)",
    "net_interchange": "market_intervals.net_interchange_mw (positive = export)",
    "available": "market_intervals.available_generation_mw (AVAILABLEGENERATION)",
    "dispatchable": "generation_intervals, dispatchable (DISPATCHABLEGENERATION)",
    "wind": "generation_intervals, wind (SS_WIND_CLEAREDMW, SS_WIND_UIGF, SS_WIND_AVAILABILITY)",
    "solar": "generation_intervals, solar (SS_SOLAR_CLEAREDMW, SS_SOLAR_UIGF, "
    "SS_SOLAR_AVAILABILITY)",
    "battery": "generation_intervals, battery (BDU_CLEAREDMW_GEN, BDU_CLEAREDMW_LOAD; "
    "bidirectional units only)",
    "flags": "market_intervals.apc_flag_bits, market_suspended_flag (DISPATCHPRICE)",
    "p5min": "aemo_demand_forecasts, p5min (P5MIN_REGIONSOLUTION)",
    "predispatch": "aemo_demand_forecasts, predispatch (PREDISPATCHREGIONSUM)",
    "gfs": "weather_forecasts (GFS via Open-Meteo Previous Runs), population-weighted, as "
    "variant C's features select it at the issue time",
    "era5": "weather_observations (ERA5 via Open-Meteo), population-weighted; published about "
    "5 days later",
    "detectors": "M4 detectors (conformal p-values against the trailing 26 weeks)",
}

ANCHOR_LABELS = {
    "direct": "none: the model forecasts demand directly",
    "latest": "the latest demand, y(o)",
    "residual": "demand at the same time yesterday, y(t − 24 h)",
    "anchored": "y(o) + y(t − 24 h) − y(o − 24 h)",
}

REQUIRED_FIELDS: tuple[str, ...] = (
    "event.start",
    "event.end",
    "event.duration_minutes",
    "event.tier",
    "event.detectors",
    "event.peak",
    "event.demand_peak",
    "event.price_lines",
    "values.demand",
    "values.demand_profile",
    "values.demand_percentile_26w",
    "values.price",
    "values.price_profile",
    "values.price_percentile_26w",
    "values.price_volatility",
    "forecasts.latest",
    "forecasts.dayahead",
    "forecasts.aemo_p5min_latest",
    "forecasts.aemo_predispatch_latest",
    "forecasts.aemo_predispatch_dayahead",
    "changes.demand_5min",
    "changes.demand_30min",
    "changes.demand_60min",
    "changes.price_5min",
    "changes.price_30min",
    "changes.price_60min",
    "supply.wind_cleared",
    "supply.wind_uigf",
    "supply.wind_available",
    "supply.wind_curtailment",
    "supply.wind_change",
    "supply.solar_cleared",
    "supply.solar_uigf",
    "supply.solar_available",
    "supply.solar_curtailment",
    "supply.solar_change",
    "supply.dispatchable",
    "supply.battery",
    "supply.net_interchange",
    "supply.available_generation",
    "supply.margin",
    "weather.gfs_temp_latest",
    "weather.gfs_temp_dayahead",
    "weather.era5_temp",
    "flags.span",
    "shap.latest",
    "shap.dayahead",
)

PairKey = tuple[int, int]  # (issue time in epoch minutes, lead in 5-minute steps)


# --------------------------------------------------------------------------- facts


def _plain(value: Any) -> Any:
    """JSON-ready copy: numpy scalars to Python, finite floats only (jsonb rejects NaN)."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if not math.isfinite(v):
            raise ValueError("a fact must not hold NaN or infinity; mark it unavailable")
        return v
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def fact(value: Any, unit: str | None, known: str, source: str, **extra: Any) -> dict[str, Any]:
    if known not in (KNOWN, RETRO):
        raise ValueError(f"unknown timing label {known!r}")
    out = {"value": _plain(value), "unit": unit, "known": known, "source": source}
    return out | {k: _plain(v) for k, v in extra.items()}


def unavailable(reason: str) -> dict[str, str]:
    if not reason:
        raise ValueError("an unavailable field needs a reason")
    return {"unavailable": reason}


def _finite(*values: float) -> bool:
    return all(v is not None and math.isfinite(float(v)) for v in values)


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True, slots=True)
class PointFact:
    """LightGBM C's stored forecast of one pair and M3's raw QR 2.5–97.5% band."""

    forecast: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class AemoRun:
    value: float  # MW: the interval (P5MIN) or the 30-minute period mean (pre-dispatch)
    available_at: pd.Timestamp  # LASTCHANGED


@dataclass(frozen=True, slots=True)
class ShapRow:
    fit: str  # retrain and band of the stored fit
    target_form: str
    anchor: float
    base: float
    forecast: float
    families: tuple[tuple[str, str, float], ...]  # (key, title, MW), catalogue order
    features: tuple[tuple[str, float], ...]  # (name, MW), column order


@dataclass(slots=True)
class EvidenceContext:
    """Every input of the evidence, aligned to ``series`` or keyed by the pairs needed."""

    series: MarketSeries
    cfg: AnomaliesConfig
    apc_bits: np.ndarray  # (n,) int
    suspended: np.ndarray  # (n,) int
    wind_available: np.ndarray  # (n,) MW, NaN before table v8 (May 2024)
    solar_available: np.ndarray
    battery_gen: np.ndarray  # (n,) MW, NaN where no row
    battery_load: np.ndarray
    point: dict[PairKey, PointFact]
    point_missing: dict[PairKey, str]
    gfs: dict[PairKey, float]
    era5: dict[int, float]  # interval index -> °C
    aemo: dict[tuple[str, int, int], AemoRun]  # (source, run minutes, target-end minutes)
    aemo_first_run: dict[str, pd.Timestamp | None]  # first loaded run per source
    shap: dict[PairKey, ShapRow]
    shap_missing: dict[PairKey, str]
    point_source: str  # e.g. "M2 lgbm_C stored forecasts, batch …; QR band, batch …"
    lay: Layout = field(init=False)
    demand_values: SlotValues = field(init=False)
    price_values: SlotValues = field(init=False)
    v: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    battery_first: int | None = field(init=False)

    def __post_init__(self) -> None:
        s, c = self.series, self.cfg
        n = len(s)
        for name in (
            "apc_bits",
            "suspended",
            "wind_available",
            "solar_available",
            "battery_gen",
            "battery_load",
        ):
            if np.asarray(getattr(self, name)).shape != (n,):
                raise ValueError(f"{name} must have one value per interval")
        self.lay = layout(s, c.profile.slot_minutes)
        self.demand_values = SlotValues(s.demand, self.lay, c.profile.days, by_day_type=True)
        self.price_values = SlotValues(s.price, self.lay, c.profile.days, by_day_type=False)
        vol = c.price_volatility
        x = price_x(s.price, c.price_level.scale_aud)
        self.v = volatility(x, vol.window_intervals)
        self.b = volatility_baseline(
            self.v, vol.baseline_days * INTERVALS_PER_DAY, vol.baseline_lag_intervals
        )
        active = np.flatnonzero(
            (np.nan_to_num(self.battery_gen) != 0) | (np.nan_to_num(self.battery_load) != 0)
        )
        self.battery_first = int(active[0]) if active.size else None


# --------------------------------------------------------------------------- pairs and peaks


def epoch_minutes(ts: pd.Timestamp) -> int:
    return int(ts.value // (60 * 10**9))


def pair_for(end: pd.Timestamp, first_lead: int) -> PairKey:
    """The issue time (on the 30-minute grid) whose lead to ``end`` lies in
    ``[first_lead, first_lead + 5]``, as D2/D3 (:func:`detectors.band_at_targets`)."""
    minutes = epoch_minutes(end)
    shifted = minutes - 5 * first_lead
    issue = shifted - shifted % 30
    return issue, (minutes - issue) // 5


def _key(p: np.ndarray, s: np.ndarray, i: int) -> tuple[float, float, int]:
    return (float(p[i]), -float(s[i]), i)


def _best_alert(event: Event, detection: Detection, names: Iterable[str]) -> tuple[str, int] | None:
    """The alert with the lowest p among ``names`` (ties: largest score, earliest, then the
    detector first in §2.3's order), as :func:`events.merge_events` picks the peak."""
    best: tuple[tuple[float, float, int], str] | None = None
    for name in ORDER:
        if name not in names or name in LINES or name not in event.detectors:
            continue
        tiers = detection.tiers[name][event.first : event.last + 1]
        rows = event.first + np.flatnonzero(tiers >= NOTABLE)
        p, s = detection.run.pvalues[name], detection.run.scores[name]
        for i in rows:
            k = _key(p, s, int(i))
            if best is None or k < best[0]:
                best = (k, name)
    return None if best is None else (best[1], best[0][2])


def peak_detector(event: Event, detection: Detection) -> str:
    """The detector holding the event's peak (a line for a lines-only event)."""
    if event.lines_only:
        return LINE_SEVERE if LINE_SEVERE in event.detectors else LINE_NOTABLE
    calibrated = [d for d in event.detectors if d not in LINES]
    found = _best_alert(event, detection, calibrated)
    if found is None or found[1] != event.peak:
        raise AssertionError("the event's peak is not its lowest-p alert")
    return found[0]


def demand_peak(event: Event, detection: Detection) -> int | None:
    """The demand peak (user 2026-11-19): the lowest-p D1–D3 alert, None if none fired."""
    found = _best_alert(event, detection, DEMAND_DETECTORS)
    return None if found is None else found[1]


def event_pairs(event: Event, detection: Detection, cfg: AnomaliesConfig) -> list[PairKey]:
    """The forecast pairs the event's evidence reads: latest and day-ahead at its peak and,
    for a demand event, at its demand peak."""
    ends = detection.run.ends
    out = []
    for idx in {event.peak, demand_peak(event, detection)} - {None}:
        assert idx is not None
        out += [
            pair_for(ends[idx], cfg.residual.short_leads[0]),
            pair_for(ends[idx], cfg.residual.dayahead_leads[0]),
        ]
    return out


def shap_pairs(event: Event, detection: Detection, cfg: AnomaliesConfig) -> list[PairKey]:
    idx = demand_peak(event, detection)
    if idx is None:
        return []
    end = detection.run.ends[idx]
    return [
        pair_for(end, cfg.residual.short_leads[0]),
        pair_for(end, cfg.residual.dayahead_leads[0]),
    ]


# --------------------------------------------------------------------------- field builders


def _nem_day(ts: pd.Timestamp) -> str:
    local = ts.tz_convert(NEM_TZ)
    return f"{local.day} {local:%b %Y %H:%M}"


def _times(ts: pd.Timestamp, tz: str) -> dict[str, str]:
    return {"local": ts.tz_convert(tz).isoformat(), "nem": ts.tz_convert(NEM_TZ).isoformat()}


def _issue_ts(minutes: int) -> pd.Timestamp:
    return pd.Timestamp(minutes * 60 * 10**9, tz="UTC")


def _event_fields(event: Event, detection: Detection, ctx: EvidenceContext) -> dict[str, Any]:
    s, tz = ctx.series, ctx.series.tz
    start = s.ends[event.first] - INTERVAL
    end = s.ends[event.last]
    src = SOURCES["detectors"]
    fired = []
    for name, min_p in zip(event.detectors, event.detector_min_p, strict=True):
        tier = int(detection.tiers[name][event.first : event.last + 1].max())
        fired.append(
            {
                "detector": name,
                "category": CATEGORY[name],
                "tier": TIER_NAMES[SEVERE if tier >= SEVERE else NOTABLE],
                "min_p": min_p,
            }
        )
    peak_name = peak_detector(event, detection)
    peak_p = None if peak_name in LINES else float(detection.run.pvalues[peak_name][event.peak])
    dfound = _best_alert(event, detection, DEMAND_DETECTORS)
    price = s.price[event.first : event.last + 1]
    lines = ctx.cfg.price_lines
    out: dict[str, Any] = {
        "start": fact(_times(start, tz), None, KNOWN, src, note="start of the first alert"),
        "end": fact(_times(end, tz), None, KNOWN, src, note="end of the last alert"),
        "duration_minutes": fact((event.last - event.first + 1) * 5, "min", KNOWN, src),
        "tier": fact(event.tier, None, KNOWN, src),
        "detectors": fact(fired, None, KNOWN, src),
        "peak": fact(
            {"interval_end": _times(s.ends[event.peak], tz), "detector": peak_name, "p": peak_p},
            None,
            KNOWN,
            src,
        ),
        "demand_peak": (
            fact(
                {"interval_end": _times(s.ends[dfound[1]], tz), "detector": dfound[0]},
                None,
                KNOWN,
                src,
            )
            if dfound is not None
            else unavailable("not a demand event: no D1–D3 alert")
        ),
        "price_lines": fact(
            {
                "intervals": int(event.last - event.first + 1),
                "at_or_above_notable": int(np.sum(price >= lines.notable_aud)),
                "at_or_above_severe": int(np.sum(price >= lines.severe_aud)),
                "notable_aud": lines.notable_aud,
                "severe_aud": lines.severe_aud,
                "max_price": float(np.nanmax(price)),
            },
            None,
            KNOWN,
            SOURCES["price"],
        ),
    }
    return out


def _profile_fact(
    values: SlotValues, idx: int, unit: str, source: str, day_type: str | None
) -> dict[str, Any]:
    block = values.at(idx)
    if block is None:
        return unavailable("the profile needs 28 local days of history")
    extra = {"day_type": day_type} if day_type else {}
    return fact(
        {
            "median": float(np.median(block)),
            "p10": float(np.percentile(block, 10)),
            "p90": float(np.percentile(block, 90)),
            "n": int(block.size),
        },
        unit,
        KNOWN,
        source,
        note="same local half-hour on the previous 28 local days",
        **extra,
    )


def _percentile_fact(values: np.ndarray, idx: int, window: int, unit: str, src: str) -> dict:
    v = values[idx]
    past = values[max(0, idx - window) : idx]
    past = past[np.isfinite(past)]
    if not _finite(v) or past.size == 0:
        return unavailable("no value at the peak or no trailing history")
    return fact(
        float(100.0 * np.mean(past <= v)),
        "%",
        KNOWN,
        src,
        n=int(past.size),
        note="share of the intervals ending in the 26 weeks before the peak at or below it",
    )


def _values_fields(event: Event, ctx: EvidenceContext) -> dict[str, Any]:
    s, idx, c = ctx.series, event.peak, ctx.cfg
    window = c.pvalues.window_weeks * WEEK_INTERVALS
    business = bool(ctx.lay.day_business[ctx.lay.local_day[idx]])
    day_type = "business days" if business else "weekend days and public holidays"
    out: dict[str, Any] = {}
    for name, unit in (("demand", "MW"), ("price", "$/MWh")):
        arr = getattr(s, name)
        out[name] = (
            fact(float(arr[idx]), unit, KNOWN, SOURCES[name])
            if _finite(arr[idx])
            else unavailable(f"no {name} value at the peak")
        )
        out[f"{name}_profile"] = _profile_fact(
            ctx.demand_values if name == "demand" else ctx.price_values,
            idx,
            unit,
            SOURCES[name],
            day_type if name == "demand" else None,
        )
        out[f"{name}_percentile_26w"] = _percentile_fact(arr, idx, window, unit, SOURCES[name])
    vol = c.price_volatility
    v, b = ctx.v[idx], ctx.b[idx]
    hour = s.price[max(0, idx - vol.window_intervals + 1) : idx + 1]
    out["price_volatility"] = (
        fact(
            {
                "ratio": float((v + vol.epsilon) / (b + vol.epsilon)),
                "rms_change": float(v),
                "baseline_median": float(b),
                "hour_min_price": float(np.nanmin(hour)),
                "hour_max_price": float(np.nanmax(hour)),
            },
            None,
            KNOWN,
            SOURCES["price"],
            note="RMS of the 5-minute changes in asinh(price/$10) over the hour to the peak, "
            "against its median over the 7 days ending 1 h before",
        )
        if _finite(v, b)
        else unavailable("the volatility baseline needs 7 days of history")
    )
    return out


def _forecast_fact(key: PairKey, ctx: EvidenceContext, actual: float, what: str) -> dict:
    pf = ctx.point.get(key)
    if pf is None:
        return unavailable(ctx.point_missing.get(key, "no stored forecast for this pair"))
    if not _finite(actual):
        return unavailable("no demand value at the peak")
    residual = actual - pf.forecast
    return fact(
        {
            "issue_time": _times(_issue_ts(key[0]), ctx.series.tz),
            "lead_minutes": 5 * key[1],
            "forecast": pf.forecast,
            "band_lower": pf.lower,
            "band_upper": pf.upper,
            "actual": actual,
            "residual": residual,
            "outside_band": bool(actual < pf.lower or actual > pf.upper),
        },
        "MW",
        KNOWN,
        ctx.point_source,
        note=f"LightGBM C's {what} forecast; band = M3's raw quantile LightGBM C "
        "2.5–97.5% (rearranged); residual = actual − forecast",
    )


def _aemo_fact(
    ctx: EvidenceContext, source: str, run: int, idx: int, peak_end: pd.Timestamp
) -> dict[str, Any]:
    s = ctx.series
    end_min = epoch_minutes(s.ends[idx])
    if source == "p5min":
        target, actual, period_end = end_min, float(s.demand[idx]), s.ends[idx]
    else:
        target = end_min + (-end_min) % 30  # the 30-minute period holding the interval
        last = idx + (target - end_min) // 5
        if last >= len(s):
            return unavailable("the pre-dispatch period ends after the loaded data")
        actual = float(np.mean(s.demand[last - 5 : last + 1]))
        period_end = s.ends[last]
    first = ctx.aemo_first_run.get(source)
    run_ts = _issue_ts(run)
    if first is None:
        return unavailable(f"AEMO's {source} forecasts are not loaded")
    if run_ts < first:
        return unavailable(
            f"AEMO's {source} forecasts are loaded from runs of {_nem_day(first)} NEM time"
        )
    hit = ctx.aemo.get((source, run, target))
    if hit is None:
        return unavailable(f"no {source} run at {run_ts.isoformat()} reaches this interval")
    known = KNOWN if hit.available_at <= peak_end and period_end <= peak_end else RETRO
    return fact(
        {
            "run_time": _times(run_ts, s.tz),
            "published": _times(hit.available_at, s.tz),
            "forecast": hit.value,
            "actual": actual,
            "error": hit.value - actual,
        },
        "MW",
        known,
        SOURCES[source],
        note="matched run: nominal run time = our issue time (M2's benchmark); "
        + ("the 5-minute interval" if source == "p5min" else "the 30-minute period mean")
        + "; error = forecast − actual",
    )


def _forecast_fields(event: Event, ctx: EvidenceContext) -> dict[str, Any]:
    s, c, idx = ctx.series, ctx.cfg, event.peak
    end = s.ends[idx]
    latest = pair_for(end, c.residual.short_leads[0])
    dayahead = pair_for(end, c.residual.dayahead_leads[0])
    actual = float(s.demand[idx])
    return {
        "latest": _forecast_fact(latest, ctx, actual, "latest"),
        "dayahead": _forecast_fact(dayahead, ctx, actual, "day-ahead"),
        "aemo_p5min_latest": _aemo_fact(ctx, "p5min", latest[0], idx, end),
        "aemo_predispatch_latest": _aemo_fact(ctx, "predispatch", latest[0], idx, end),
        "aemo_predispatch_dayahead": _aemo_fact(ctx, "predispatch", dayahead[0], idx, end),
    }


def _change_fact(values: np.ndarray, idx: int, steps: int, unit: str, src: str) -> dict:
    if idx - steps < 0 or not _finite(values[idx], values[idx - steps]):
        return unavailable(f"no value {steps * 5} min before the peak")
    a, b = float(values[idx - steps]), float(values[idx])
    return fact({"from": a, "to": b, "change": b - a, "minutes": steps * 5}, unit, KNOWN, src)


def _changes_fields(event: Event, ctx: EvidenceContext) -> dict[str, Any]:
    s, idx = ctx.series, event.peak
    out = {}
    for minutes in ctx.cfg.evidence.change_minutes:
        for name, unit in (("demand", "MW"), ("price", "$/MWh")):
            out[f"{name}_{minutes}min"] = _change_fact(
                getattr(s, name), idx, minutes // 5, unit, SOURCES[name]
            )
    return out


def _mw(values: np.ndarray, idx: int, src: str, missing: str, **extra: Any) -> dict:
    v = values[idx]
    return fact(float(v), "MW", KNOWN, src, **extra) if _finite(v) else unavailable(missing)


def _supply_fields(event: Event, ctx: EvidenceContext) -> dict[str, Any]:
    s, idx, c = ctx.series, event.peak, ctx.cfg
    steps = c.evidence.supply_change_minutes // 5
    min_uigf = c.isolation_forest.curtailment_min_uigf_mw
    out: dict[str, Any] = {}
    for kind in ("wind", "solar"):
        cleared = getattr(s, f"{kind}_cleared")
        uigf = getattr(s, f"{kind}_uigf")
        avail = getattr(ctx, f"{kind}_available")
        src = SOURCES[kind]
        out[f"{kind}_cleared"] = _mw(cleared, idx, src, f"no {kind} output at the peak")
        out[f"{kind}_uigf"] = _mw(uigf, idx, src, f"no {kind} UIGF at the peak")
        out[f"{kind}_available"] = _mw(
            avail,
            idx,
            src,
            "AEMO publishes semi-scheduled availability from May 2024 (DISPATCHREGIONSUM v8)",
        )
        if _finite(cleared[idx], uigf[idx]) and uigf[idx] >= min_uigf:
            out[f"{kind}_curtailment"] = fact(
                float(1.0 - cleared[idx] / uigf[idx]),
                "share",
                KNOWN,
                src,
                note="1 − cleared / UIGF (against UIGF: m4_plan.md §1, step 1b)",
            )
        else:
            out[f"{kind}_curtailment"] = unavailable(
                f"{kind} UIGF below {min_uigf:g} MW at the peak"
            )
        out[f"{kind}_change"] = _change_fact(cleared, idx, steps, "MW", src)
    out["dispatchable"] = _mw(s.dispatchable, idx, SOURCES["dispatchable"], "no value")
    first = ctx.battery_first
    if first is None or idx < first:
        out["battery"] = unavailable(
            "no non-zero bidirectional-unit (battery) interval at or before the peak; "
            "batteries outside that category sit in the dispatchable total"
        )
    elif _finite(ctx.battery_gen[idx], ctx.battery_load[idx]):
        out["battery"] = fact(
            {
                "generation": float(ctx.battery_gen[idx]),
                "load": float(ctx.battery_load[idx]),
            },
            "MW",
            KNOWN,
            SOURCES["battery"],
            note="bidirectional units only",
        )
    else:
        out["battery"] = unavailable("no bidirectional-unit row at the peak")
    ni = s.net_interchange[idx]
    out["net_interchange"] = (
        fact(
            {"mw": float(ni), "direction": "export" if ni > 0 else "import" if ni < 0 else "none"},
            "MW",
            KNOWN,
            SOURCES["net_interchange"],
        )
        if _finite(ni)
        else unavailable("no net interchange at the peak")
    )
    out["available_generation"] = _mw(s.available, idx, SOURCES["available"], "no value")
    margin = s.available[idx] - s.demand[idx]
    out["margin"] = (
        fact(float(margin), "MW", KNOWN, SOURCES["available"], note="available − demand")
        if _finite(margin)
        else unavailable("no available generation or demand at the peak")
    )
    return out


def _weather_fields(event: Event, ctx: EvidenceContext) -> dict[str, Any]:
    end, c = ctx.series.ends[event.peak], ctx.cfg
    out: dict[str, Any] = {}
    for name, first in (
        ("latest", c.residual.short_leads[0]),
        ("dayahead", c.residual.dayahead_leads[0]),
    ):
        key = pair_for(end, first)
        v = ctx.gfs.get(key)
        out[f"gfs_temp_{name}"] = (
            fact(
                v, "°C", KNOWN, SOURCES["gfs"], issue_time=_times(_issue_ts(key[0]), ctx.series.tz)
            )
            if v is not None and _finite(v)
            else unavailable(
                "no GFS run published by the issue time covers the target (e.g. the "
                "Dec 2023 – Jan 2024 archive gap)"
            )
        )
    era5 = ctx.era5.get(event.peak)
    out["era5_temp"] = (
        fact(era5, "°C", RETRO, SOURCES["era5"])
        if era5 is not None and _finite(era5)
        else unavailable("no ERA5 value at the peak")
    )
    return out


def _flags_fields(event: Event, ctx: EvidenceContext) -> dict[str, Any]:
    bits = np.asarray(ctx.apc_bits[event.first : event.last + 1], dtype=np.int64)
    susp = np.asarray(ctx.suspended[event.first : event.last + 1], dtype=np.int64)
    by_bit = [
        {"bit": b, "meaning": APC_SHORT[b], "intervals": int(np.sum((bits & b) != 0))}
        for b in sorted(APC_SHORT)
        if np.any((bits & b) != 0)
    ]
    return {
        "span": fact(
            {
                "intervals": len(bits),
                "flagged_intervals": int(np.sum((bits != 0) | (susp != 0))),
                "apc_bits": by_bit,
                "suspended_intervals": int(np.sum(susp != 0)),
            },
            None,
            KNOWN,
            SOURCES["flags"],
            note="evidence only; the flags are never detector inputs",
        )
    }


def _shap_fact(key: PairKey, ctx: EvidenceContext, idx: int, what: str) -> dict[str, Any]:
    row = ctx.shap.get(key)
    if row is None:
        return unavailable(ctx.shap_missing.get(key, "no explanation for this pair"))
    return fact(
        {
            "interval_end": _times(ctx.series.ends[idx], ctx.series.tz),
            "issue_time": _times(_issue_ts(key[0]), ctx.series.tz),
            "lead_minutes": 5 * key[1],
            "fit": row.fit,
            "target_form": row.target_form,
            "anchor": {"label": ANCHOR_LABELS[row.target_form], "mw": row.anchor},
            "base_value": row.base,
            "forecast": row.forecast,
            "families": [{"key": k, "title": t, "mw": v} for k, t, v in row.families],
            "features": {n: v for n, v in row.features},
        },
        "MW",
        KNOWN,
        f"LightGBM TreeSHAP (pred_contrib) of M2's stored fit {row.fit}",
        note=f"how LightGBM C arrived at its {what} forecast: forecast = anchor + base value "
        "+ Σ contributions. Contributions describe the model's arithmetic, not the grid's "
        "physics; correlated features share credit",
    )


def _shap_fields(event: Event, detection: Detection, ctx: EvidenceContext) -> dict[str, Any]:
    idx = demand_peak(event, detection)
    if idx is None:
        reason = "not a demand event: no D1–D3 alert"
        return {"latest": unavailable(reason), "dayahead": unavailable(reason)}
    end, c = ctx.series.ends[idx], ctx.cfg
    return {
        "latest": _shap_fact(pair_for(end, c.residual.short_leads[0]), ctx, idx, "latest"),
        "dayahead": _shap_fact(pair_for(end, c.residual.dayahead_leads[0]), ctx, idx, "day-ahead"),
    }


def build_evidence(event: Event, detection: Detection, ctx: EvidenceContext) -> dict[str, Any]:
    """Every §2.5 field of one event."""
    return {
        "version": VERSION,
        "event": _event_fields(event, detection, ctx),
        "values": _values_fields(event, ctx),
        "forecasts": _forecast_fields(event, ctx),
        "changes": _changes_fields(event, ctx),
        "supply": _supply_fields(event, ctx),
        "weather": _weather_fields(event, ctx),
        "flags": _flags_fields(event, ctx),
        "shap": _shap_fields(event, detection, ctx),
    }


# --------------------------------------------------------------------------- A4 checks


def _get(evidence: Mapping[str, Any], path: str) -> Any:
    node: Any = evidence
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def missing_fields(evidence: Mapping[str, Any]) -> list[str]:
    """A4's completeness check: required fields that are neither a labelled fact nor
    explicitly unavailable with a reason."""
    bad = []
    for path in REQUIRED_FIELDS:
        node = _get(evidence, path)
        if not isinstance(node, Mapping):
            bad.append(path)
        elif "unavailable" in node:
            if not isinstance(node["unavailable"], str) or not node["unavailable"].strip():
                bad.append(path)
        elif node.get("value") is None or node.get("known") not in (KNOWN, RETRO):
            bad.append(path)
    return bad


def banned_in(text: str, phrases: Sequence[str]) -> list[str]:
    """Banned phrases found in ``text`` (case-insensitive, on word boundaries)."""
    return [p for p in phrases if re.search(rf"\b{re.escape(p)}\b", text, flags=re.IGNORECASE)]


# --------------------------------------------------------------------------- summaries


def _value(evidence: Mapping[str, Any], path: str) -> Any:
    node = _get(evidence, path)
    return node.get("value") if isinstance(node, Mapping) else None


def _local(iso: str) -> pd.Timestamp:
    return pd.Timestamp(iso)


def _hm(ts: pd.Timestamp) -> str:
    return f"{ts:%H:%M}"


def _day(ts: pd.Timestamp) -> str:
    return f"{ts:%a} {ts.day} {ts:%b %Y}"


def _mw_text(v: float) -> str:
    return f"{v:,.0f} MW"


def _money(v: float, digits: int = 0) -> str:
    """``money`` without a negative zero (−$0.30 rounds to $0)."""
    r = round(float(v), digits)
    return money(0.0 if r == 0 else r, digits)


def _aud(v: float, digits: int = 0) -> str:
    return f"{_money(v, digits)}/MWh"


def _price_span(lo: float, hi: float) -> str:
    """The hour's price range, in whole dollars unless that would hide it."""
    if round(lo, 2) == round(hi, 2):
        digits = 0 if round(lo) == round(lo, 2) else 2
        exact = "" if lo == hi else " (to the cent)"
        return f"the price stayed at {_aud(lo, digits)}{exact}"
    digits = 2 if round(lo) == round(hi) else 0
    return f"the price ranged from {_money(lo, digits)} to {_aud(hi, digits)}"


def _span(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if start.date() == end.date():
        return f"{_hm(start)}–{_hm(end)} {_day(start)}"
    return f"{_hm(start)} {_day(start)} – {_hm(end)} {_day(end)}"


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _p_text(p: float) -> str:
    """0.0042, or 1.9 × 10⁻⁵ below 0.001."""
    if p >= 0.001:
        return f"{p:.2g}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"{mantissa} × 10{str(int(exponent)).translate(_SUPERSCRIPT)}"


def _lead_sentence(evidence: Mapping[str, Any]) -> str:
    peak = _value(evidence, "event.peak")
    detector = peak["detector"]
    at = _hm(_local(peak["interval_end"]["local"]))
    demand, price = _value(evidence, "values.demand"), _value(evidence, "values.price")
    if detector == DEMAND_LEVEL:
        prof = _get(evidence, "values.demand_profile") or {}
        text = f"demand in the 5 minutes to {at} was {_mw_text(demand)}"
        if "value" in prof:
            v = prof["value"]
            text += (
                f"; the median for that half-hour on the {prof['day_type']} among the previous "
                f"28 days was {_mw_text(v['median'])} (10th–90th percentile "
                f"{v['p10']:,.0f}–{v['p90']:,.0f} MW)"
            )
        return text
    if detector in (RESIDUAL_SHORT, RESIDUAL_DAYAHEAD):
        what = "latest" if detector == RESIDUAL_SHORT else "dayahead"
        f = _value(evidence, f"forecasts.{what}")
        if f is None:
            return f"demand in the 5 minutes to {at} was {_mw_text(demand)}"
        issued = _local(f["issue_time"]["local"])
        when = (
            f"issued at {_hm(issued)}"
            if detector == RESIDUAL_SHORT
            else f"issued at {_hm(issued)} on {_day(issued)}"
        )
        side = "above" if f["residual"] > 0 else "below"
        inside = "outside" if f["outside_band"] else "inside"
        label = "latest" if detector == RESIDUAL_SHORT else "day-ahead"
        return (
            f"demand in the 5 minutes to {at} was {_mw_text(f['actual'])}, "
            f"{abs(f['residual']):,.0f} MW {side} LightGBM C's {label} forecast "
            f"({_mw_text(f['forecast'])}, {when}) and {inside} its 2.5–97.5% quantile band "
            f"({f['band_lower']:,.0f}–{f['band_upper']:,.0f} MW)"
        )
    if detector == PRICE_LEVEL:
        prof = _get(evidence, "values.price_profile") or {}
        text = f"the price in the 5 minutes to {at} was {_aud(price)}"
        if "value" in prof:
            v = prof["value"]
            text += (
                f"; the median for that half-hour on the previous 28 days was "
                f"{_aud(v['median'])} (10th–90th percentile {_money(v['p10'])} to "
                f"{_aud(v['p90'])})"
            )
        return text
    if detector == PRICE_VOLATILITY:
        v = _value(evidence, "values.price_volatility")
        if v is None:
            return f"the price in the 5 minutes to {at} was {_aud(price)}"
        span = _price_span(v["hour_min_price"], v["hour_max_price"])
        return (
            f"in the hour to {at}, the root-mean-square 5-minute change in asinh(price/$10) "
            f"was {v['ratio']:.2g} times its median over the previous 7 days, and {span}"
        )
    if detector == ISOLATION_FOREST:
        return (
            f"the multivariate detector (Isolation Forest) gave the 5 minutes to {at} a "
            f"p-value of {_p_text(peak['p'])} against its training window (the 26 weeks "
            f"before the month)"
        )
    lines = _value(evidence, "event.price_lines")
    n, k = lines["intervals"], lines["at_or_above_notable"]
    line = _aud(lines["notable_aud"])
    if n == 1:
        share = f"the event's single interval was at or above {line}"
    else:
        share = f"{k} of the event's {n} intervals {'was' if k == 1 else 'were'} at or above {line}"
    text = f"the price reached {_aud(lines['max_price'])} in the 5 minutes to {at}; {share}"
    if lines["at_or_above_severe"] and n > 1:
        text += f" and {lines['at_or_above_severe']} at or above {_aud(lines['severe_aud'])}"
    return text


def _move_sentence(evidence: Mapping[str, Any], detector: str) -> str:
    name = "price" if detector in DEMAND_DETECTORS else "demand"
    ch = _get(evidence, f"changes.{name}_5min") or {}
    if "value" not in ch:
        return f"The {name} 5 minutes before is not available."
    a, b = ch["value"]["from"], ch["value"]["to"]
    if name == "price":
        noun, start, stop = "the price", _money(a), _aud(b)
    else:
        noun, start, stop = "demand", f"{a:,.0f}", _mw_text(b)
    if a == b:
        return f"Over the same 5 minutes, {noun} stayed at {stop}."
    return f"Over the same 5 minutes, {noun} moved from {start} to {stop}."


def _flags_sentence(evidence: Mapping[str, Any]) -> str:
    f = _value(evidence, "flags.span")
    if not f["flagged_intervals"]:
        return "AEMO flagged nothing in the event's span."
    parts = [f"{b['meaning']} in {b['intervals']}" for b in f["apc_bits"]]
    if f["suspended_intervals"]:
        parts.append(f"the market suspended in {f['suspended_intervals']}")
    return (
        f"In the event's span, AEMO flagged {f['flagged_intervals']} of {f['intervals']} "
        f"intervals: {'; '.join(parts)}."
    )


def summarise(evidence: Mapping[str, Any]) -> str:
    """Three facts-only sentences (user 2026-11-19), from the evidence alone."""
    start = _local(_value(evidence, "event.start")["local"])
    end = _local(_value(evidence, "event.end")["local"])
    tier = str(_value(evidence, "event.tier")).capitalize()
    detector = _value(evidence, "event.peak")["detector"]
    first = f"{tier} event, {_span(start, end)} (Melbourne time): {_lead_sentence(evidence)}."
    return " ".join([first, _move_sentence(evidence, detector), _flags_sentence(evidence)])


def check_summary(text: str, cfg: AnomaliesConfig) -> list[str]:
    return banned_in(text, cfg.evidence.banned_phrases)
