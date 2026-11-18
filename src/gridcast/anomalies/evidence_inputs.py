"""Load what the evidence of a split's events needs (m4_plan.md §2.5).

Only the pairs and intervals the events need are looked up, and every load can end at an
interval (``last_end``), so the warm-up smoke run never reads data after its warm-up.

* **Supply and flags:** wind and solar availability, battery output (``generation_intervals``)
  and the operator flags (``market_intervals``), aligned to the series.
* **Forecasts:** LightGBM C's stored forecasts, M3's raw QR 2.5–97.5% band and M2's common
  mask. Development reads run 2 and QR batch run 85; the holdout adds ``holdout_main`` and
  the holdout QR batch from the holdout's first issue time, because a holdout event's
  day-ahead pair can be issued in late September 2025, which only run 2 forecast.
* **Weather:** the GFS temperature that variant C's feature row selects at each pair's issue
  time, and ERA5 at the peak, from M2's feature builder.
* **AEMO:** the matched P5MIN and pre-dispatch runs (nominal run time = our issue time).
* **SHAP** at every demand peak, through :func:`gridcast.explainability.pairs.explain_pairs`.
  E1's gate must pass, or :class:`~gridcast.explainability.treeshap.GateError` stops the run
  (explanations withheld, §2.6).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from gridcast.anomalies.events import Event
from gridcast.anomalies.evidence import (
    AemoRun,
    EvidenceContext,
    PairKey,
    PointFact,
    ShapRow,
    build_evidence,
    event_pairs,
    shap_pairs,
    summarise,
)
from gridcast.anomalies.run import Detection
from gridcast.anomalies.series import MarketSeries
from gridcast.config import AppConfig
from gridcast.database.queries import load_market_frame
from gridcast.explainability.families import families
from gridcast.explainability.pairs import ModelCache, explain_pairs
from gridcast.explainability.stored import StoredFit, split_period, stored_fits
from gridcast.explainability.treeshap import GateResult, empty_gate, require
from gridcast.features.builder import FeatureBuilder
from gridcast.features.calendar import load_calendar_days
from gridcast.features.inputs import load_weather_inputs
from gridcast.features.weather import select_observed
from gridcast.forecasting.origins import ForecastSetup, build_context
from gridcast.intervals.cqr_batch import read_qr_quantiles
from gridcast.intervals.inputs import read_wide
from gridcast.timeutils import NEM_TZ

EXTRA_SQL = """
SELECT m.interval_end, m.apc_flag_bits, m.market_suspended_flag,
       w.available_mw AS wind_available, s.available_mw AS solar_available,
       b.cleared_mw AS battery_gen, b.load_mw AS battery_load
FROM market_intervals m
LEFT JOIN generation_intervals w
       ON w.region_id = m.region_id AND w.interval_end = m.interval_end AND w.kind = 'wind'
LEFT JOIN generation_intervals s
       ON s.region_id = m.region_id AND s.interval_end = m.interval_end AND s.kind = 'solar'
LEFT JOIN generation_intervals b
       ON b.region_id = m.region_id AND b.interval_end = m.interval_end AND b.kind = 'battery'
WHERE m.region_id = :region_id AND m.interval_end <= :last_end
ORDER BY m.interval_end
"""

AEMO_SQL = """
SELECT source, run_time, target_end, total_demand_mw, available_at
FROM aemo_demand_forecasts
WHERE region_id = :region_id AND run_time = ANY(:runs) AND target_end = ANY(:targets)
"""

# The loaded range is metadata for the "unavailable" reasons, not an evidence value.
FIRST_RUN_SQL = """
SELECT source, min(run_time) AS first FROM aemo_demand_forecasts
WHERE region_id = :region_id GROUP BY source
"""

OUTSIDE_MASK = (
    "outside M2's common mask: a model's forecast or the actual is missing for this pair "
    "(e.g. no GFS forecast weather in the Dec 2023 – Jan 2024 archive gap)"
)


def _minutes(times: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(times.as_unit("ns").asi8 // (60 * 10**9), dtype=np.int64)


def _ts(minutes: int) -> pd.Timestamp:
    return pd.Timestamp(minutes * 60 * 10**9, tz="UTC")


# --------------------------------------------------------------------------- supply, flags


@dataclass(slots=True)
class Extras:
    apc_bits: np.ndarray
    suspended: np.ndarray
    wind_available: np.ndarray
    solar_available: np.ndarray
    battery_gen: np.ndarray
    battery_load: np.ndarray


def load_extras(engine: Engine, region_id: str, series: MarketSeries) -> Extras:
    with engine.connect() as conn:
        frame = pd.read_sql(
            text(EXTRA_SQL), conn, params={"region_id": region_id, "last_end": series.ends[-1]}
        )
    ends = pd.DatetimeIndex(pd.to_datetime(frame["interval_end"], utc=True))
    if not ends.equals(series.ends):
        raise ValueError("market_intervals rows differ from the detector series' grid")

    def col(name: str) -> np.ndarray:
        return frame[name].to_numpy(dtype=np.float64, na_value=np.nan)

    return Extras(
        apc_bits=frame["apc_flag_bits"].fillna(0).to_numpy(dtype=np.int64),
        suspended=frame["market_suspended_flag"].fillna(0).to_numpy(dtype=np.int64),
        wind_available=col("wind_available"),
        solar_available=col("solar_available"),
        battery_gen=col("battery_gen"),
        battery_load=col("battery_load"),
    )


# --------------------------------------------------------------------------- forecasts


@dataclass(frozen=True, slots=True)
class PointSource:
    """One stored M2 batch and its QR band, used for issue times in [start, end)."""

    point_batch: str
    qr_batch: str
    start: pd.Timestamp
    end: pd.Timestamp | None


def point_sources(cfg: AppConfig, split: str) -> list[PointSource]:
    ex, res = cfg.anomalies.explain, cfg.anomalies.residual
    dev = PointSource(ex.dev_batch, res.dev_qr_batch, *split_period(cfg, "dev"))
    if split == "dev":
        return [dev]
    if split == "holdout":
        return [dev, PointSource(ex.holdout_batch, res.holdout_qr_batch, *split_period(cfg, split))]
    raise ValueError(f"unknown split {split!r}")


def read_common_mask(directory: Path, batch: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """M2's scored pairs of a stored batch (every model's forecast and the actual finite),
    read one file at a time."""
    files = sorted(directory.glob(f"demand__*__{batch}.parquet"))
    if not files:
        raise FileNotFoundError(f"no stored forecasts of batch {batch} in {directory}")
    times: pd.DatetimeIndex | None = None
    mask: np.ndarray | None = None
    for path in files:
        t, values = read_wide(path)
        if times is None:
            times, mask = t, np.isfinite(values)
        elif not times.equals(t):
            raise ValueError(f"{path.name}: issue times differ from the batch's other files")
        else:
            assert mask is not None
            mask &= np.isfinite(values)
    assert times is not None and mask is not None
    return times, mask


def lookup_points(
    cfg: AppConfig, split: str, keys: Sequence[PairKey]
) -> tuple[dict[PairKey, PointFact], dict[PairKey, str], str]:
    """Stored forecast and band of each needed pair, or the reason it has none."""
    ex, res = cfg.anomalies.explain, cfg.anomalies.residual
    backtests = cfg.data_path(cfg.paths.backtest_dir)
    found: dict[PairKey, PointFact] = {}
    missing: dict[PairKey, str] = {}
    keys = sorted(set(keys))
    labels = []
    first_issue: pd.Timestamp | None = None
    for src in point_sources(cfg, split):
        times, mask = read_common_mask(backtests, src.point_batch)
        p_times, point = read_wide(backtests / f"demand__{ex.model}__{src.point_batch}.parquet")
        q_times, q = read_qr_quantiles(
            cfg.data_path(cfg.intervals.output_dir) / src.qr_batch,
            [res.lower_quantile, res.upper_quantile],
        )
        if not p_times.equals(times):
            raise ValueError(f"batch {src.point_batch}: {ex.model} issue times differ")
        rows_in_qr = q_times.get_indexer(times)
        if np.any(rows_in_qr < 0):
            raise ValueError(f"QR batch {src.qr_batch} lacks issue times of {src.point_batch}")
        lower = q[res.lower_quantile][rows_in_qr]
        upper = q[res.upper_quantile][rows_in_qr]
        valid = mask & np.isfinite(lower) & np.isfinite(upper)
        if np.any(mask & ~valid):
            raise ValueError(f"QR batch {src.qr_batch} lacks quantiles on common-mask pairs")
        minutes = _minutes(times)
        first_issue = times[0] if first_issue is None else min(first_issue, times[0])
        labels.append(f"M2 {ex.model} batch {src.point_batch}, M3 QR batch {src.qr_batch}")
        lo = int(src.start.value // (60 * 10**9))
        hi = None if src.end is None else int(src.end.value // (60 * 10**9))
        for key in keys:
            issue, lead = key
            if issue < lo or (hi is not None and issue >= hi) or key in found:
                continue
            row = int(np.searchsorted(minutes, issue))
            if row >= len(minutes) or minutes[row] != issue:
                continue
            if lead > point.shape[1] or not valid[row, lead - 1]:
                missing[key] = OUTSIDE_MASK
                continue
            found[key] = PointFact(
                float(np.float32(point[row, lead - 1])),
                float(lower[row, lead - 1]),
                float(upper[row, lead - 1]),
            )
    for key in keys:
        if key in found or key in missing:
            continue
        when = _ts(key[0])
        if first_issue is not None and when < first_issue:
            missing[key] = (
                "LightGBM C's stored forecasts start at issue time "
                f"{first_issue.tz_convert(NEM_TZ):%Y-%m-%d %H:%M} NEM time"
            )
        else:
            missing[key] = "no stored forecast at this issue time"
    return found, missing, "; ".join(labels)


# --------------------------------------------------------------------------- builder


def truncated_builder(
    engine: Engine, cfg: AppConfig, region_id: str, last_end: pd.Timestamp | None
) -> FeatureBuilder:
    """M2's feature builder on market data ending at ``last_end`` (None: everything)."""
    region = cfg.region(region_id)
    market = load_market_frame(engine, region_id)
    if last_end is not None:
        market = market[market["interval_end"] <= last_end]
    with engine.connect() as conn:
        calendar = load_calendar_days(conn, region.holiday_subdiv)
    ctx = build_context(market, "total_demand_mw", region.local_tz, calendar)
    weather = load_weather_inputs(engine, cfg, region_id, ctx.times)
    return FeatureBuilder(ctx, ForecastSetup.from_config(cfg), weather, cfg.features.degree_base_c)


def gfs_temperatures(
    builder: FeatureBuilder, variant: str, keys: Sequence[PairKey]
) -> dict[PairKey, float]:
    """``wx_temp`` of variant ``variant``'s feature row at each pair (NaN: none selected)."""
    from gridcast.explainability.sample import feature_rows, origin_positions
    from gridcast.features import catalogue

    keys = sorted(set(keys))
    if not keys:
        return {}
    issue = pd.DatetimeIndex([_ts(k[0]) for k in keys])
    leads = np.array([k[1] for k in keys], dtype=np.int64)
    X = feature_rows(builder, variant, origin_positions(builder, issue), leads)
    col = catalogue.feature_names(variant).index("wx_temp")
    return {k: float(X[i, col]) for i, k in enumerate(keys)}


def era5_temperatures(
    builder: FeatureBuilder, series: MarketSeries, indices: Sequence[int]
) -> dict[int, float]:
    if builder.weather is None:
        raise ValueError("the builder has no weather")
    idx = sorted(set(indices))
    if not idx:
        return {}
    pos = np.asarray(builder.ctx.times.get_indexer(series.ends[idx]), dtype=np.int64)
    if np.any(pos < 0):
        raise LookupError("a peak interval is not on the builder's grid")
    temp = select_observed(builder.weather, ["temp_weighted"], pos)["temp_weighted"]
    return {i: float(v) for i, v in zip(idx, temp, strict=True) if np.isfinite(v)}


# --------------------------------------------------------------------------- AEMO


def lookup_aemo(
    engine: Engine,
    region_id: str,
    runs: Sequence[int],
    targets: Sequence[int],
    last_end: pd.Timestamp,
) -> tuple[dict[tuple[str, int, int], AemoRun], dict[str, pd.Timestamp | None]]:
    run_ts = [_ts(m).to_pydatetime() for m in sorted(set(runs))]
    target_ts = [_ts(m).to_pydatetime() for m in sorted(set(targets))]
    with engine.connect() as conn:
        first = {
            str(r.source): pd.Timestamp(r.first).tz_convert("UTC")
            for r in conn.execute(text(FIRST_RUN_SQL), {"region_id": region_id})
        }
        frame = (
            pd.read_sql(
                text(AEMO_SQL),
                conn,
                params={"region_id": region_id, "runs": run_ts, "targets": target_ts},
            )
            if run_ts
            else pd.DataFrame()
        )
    out: dict[tuple[str, int, int], AemoRun] = {}
    if frame.empty:
        return out, {s: first.get(s) for s in ("p5min", "predispatch")}
    runs_min = _minutes(pd.DatetimeIndex(pd.to_datetime(frame["run_time"], utc=True)))
    target_idx = pd.DatetimeIndex(pd.to_datetime(frame["target_end"], utc=True))
    published = pd.DatetimeIndex(pd.to_datetime(frame["available_at"], utc=True))
    keep = np.asarray(target_idx <= last_end)
    values = frame["total_demand_mw"].to_numpy(dtype=np.float64)
    sources = frame["source"].astype(str).to_numpy()
    targets_min = _minutes(target_idx)
    for i in np.flatnonzero(keep):
        key = (str(sources[i]), int(runs_min[i]), int(targets_min[i]))
        out[key] = AemoRun(float(values[i]), pd.Timestamp(published[int(i)]))
    return out, {s: first.get(s) for s in ("p5min", "predispatch")}


# --------------------------------------------------------------------------- SHAP


def explain_peaks(
    builder: FeatureBuilder,
    fits: Sequence[StoredFit],
    keys: Sequence[PairKey],
    point: dict[PairKey, PointFact],
    missing: dict[PairKey, str],
    cfg: AppConfig,
) -> tuple[dict[PairKey, ShapRow], dict[PairKey, str], GateResult]:
    """SHAP of every demand-peak pair with a stored forecast; E1 must pass."""
    keys = sorted(set(keys))
    have = [k for k in keys if k in point]
    out_missing = {
        k: missing.get(k, "no stored forecast for this pair") for k in keys if k not in point
    }
    if not have:
        return {}, out_missing, empty_gate(cfg.anomalies.explain.sum_tolerance)
    issue = pd.DatetimeIndex([_ts(k[0]) for k in have])
    leads = np.array([k[1] for k in have], dtype=np.int64)
    stored = np.array([point[k].forecast for k in have], dtype=np.float32)
    res = explain_pairs(builder, fits, issue, leads, stored, cfg, ModelCache(cfg))
    require(res.gate, "the demand-peak explanations")
    ex = res.explanation
    assert ex is not None
    fam = families(cfg.anomalies.explain.variant)
    sums = fam.sums(ex.contributions)
    rows = {}
    for i, k in enumerate(have):
        fit = res.fits[int(res.fit_index[i])]
        rows[k] = ShapRow(
            fit=fit.key,
            target_form=fit.target_form,
            anchor=float(ex.anchor[i]),
            base=float(ex.base[i]),
            forecast=float(ex.forecast[i]),
            families=tuple(
                (key, title, float(sums[i, j]))
                for j, (key, title) in enumerate(zip(fam.keys, fam.titles, strict=True))
            ),
            features=tuple(
                (name, float(ex.contributions[i, j])) for j, name in enumerate(ex.columns)
            ),
        )
    return rows, out_missing, res.gate


# --------------------------------------------------------------------------- the context


@dataclass(slots=True)
class EvidenceRun:
    context: EvidenceContext
    shap_gate: GateResult
    evidence: list[dict]
    summaries: list[str]


def explain_fits(engine: Engine, cfg: AppConfig, split: str) -> list[StoredFit]:
    fits = stored_fits(engine, cfg, "dev")
    if split == "holdout":
        fits += stored_fits(engine, cfg, "holdout")
    return fits


def build_evidence_context(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    split: str,
    series: MarketSeries,
    detection: Detection,
    events: Sequence[Event],
) -> tuple[EvidenceContext, GateResult]:
    """Everything :func:`~gridcast.anomalies.evidence.build_evidence` reads for ``events``."""
    a = cfg.anomalies
    last_end = series.ends[-1]
    extras = load_extras(engine, region_id, series)
    pairs = sorted({k for e in events for k in event_pairs(e, detection, a)})
    peaks = sorted({k for e in events for k in shap_pairs(e, detection, a)})
    point, missing, label = lookup_points(cfg, split, [*pairs, *peaks])
    builder = truncated_builder(engine, cfg, region_id, last_end)
    gfs = gfs_temperatures(builder, a.explain.variant, pairs)
    era5 = era5_temperatures(builder, series, [e.peak for e in events])
    runs = [k[0] for k in pairs]
    targets = []
    for e in events:
        end_min = int(series.ends[e.peak].value // (60 * 10**9))
        targets += [end_min, end_min + (-end_min) % 30]
    aemo, first = lookup_aemo(engine, region_id, runs, targets, last_end)
    shap, shap_missing, gate = explain_peaks(
        builder, explain_fits(engine, cfg, split), peaks, point, missing, cfg
    )
    ctx = EvidenceContext(
        series=series,
        cfg=a,
        apc_bits=extras.apc_bits,
        suspended=extras.suspended,
        wind_available=extras.wind_available,
        solar_available=extras.solar_available,
        battery_gen=extras.battery_gen,
        battery_load=extras.battery_load,
        point=point,
        point_missing=missing,
        gfs=gfs,
        era5=era5,
        aemo=aemo,
        aemo_first_run=first,
        shap=shap,
        shap_missing=shap_missing,
        point_source=label,
    )
    return ctx, gate


def run_evidence(
    engine: Engine,
    cfg: AppConfig,
    region_id: str,
    split: str,
    series: MarketSeries,
    detection: Detection,
    events: Sequence[Event] | None = None,
) -> EvidenceRun:
    """Evidence and summaries of ``events`` (default: every event of the detection)."""
    events = list(detection.events if events is None else events)
    ctx, gate = build_evidence_context(engine, cfg, region_id, split, series, detection, events)
    evidence = [build_evidence(e, detection, ctx) for e in events]
    return EvidenceRun(ctx, gate, evidence, [summarise(ev) for ev in evidence])
