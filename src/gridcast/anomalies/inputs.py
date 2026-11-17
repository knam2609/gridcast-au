"""Load the detectors' inputs: market and supply totals, and M3's stored quantile band.

* :func:`load_market_series` joins ``market_intervals`` with ``generation_intervals``
  (dispatchable, wind, solar) and the day type from ``calendar_days``. A business day is a
  local date that is neither a weekend nor a VIC public holiday, by interval start, as M2's
  calendar features.
* :func:`read_band` reads the rearranged ``q025``/``q975`` of a stored QR batch and the
  common mask of the M2 batch whose pairs it scores (run 2 for development). It checks
  that both cover the same issue times and that the QR batch's actuals equal run 2's.
* :func:`check_band_actuals` checks the stored actuals against market demand, so D2/D3
  score the same demand the other detectors see.

Every load can end at an interval (``last_end``), so the warm-up smoke run never reads
data after the warm-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from gridcast.anomalies.series import MarketSeries, QuantileBand, epoch_minutes
from gridcast.config import AppConfig
from gridcast.features.calendar import load_calendar_days
from gridcast.intervals.cqr_batch import QR_RUN_NAME, read_qr_quantiles
from gridcast.intervals.inputs import read_stored_batch, read_wide
from gridcast.timeutils import INTERVAL

MARKET_SQL = """
SELECT m.interval_end,
       m.total_demand_mw   AS demand,
       m.rrp_aud_mwh       AS price,
       m.net_interchange_mw AS net_interchange,
       m.available_generation_mw AS available,
       d.cleared_mw        AS dispatchable,
       w.cleared_mw        AS wind_cleared,
       w.uigf_mw           AS wind_uigf,
       s.cleared_mw        AS solar_cleared,
       s.uigf_mw           AS solar_uigf
FROM market_intervals m
LEFT JOIN generation_intervals d
       ON d.region_id = m.region_id AND d.interval_end = m.interval_end AND d.kind = 'dispatchable'
LEFT JOIN generation_intervals w
       ON w.region_id = m.region_id AND w.interval_end = m.interval_end AND w.kind = 'wind'
LEFT JOIN generation_intervals s
       ON s.region_id = m.region_id AND s.interval_end = m.interval_end AND s.kind = 'solar'
WHERE m.region_id = :region_id AND m.interval_end <= :last_end
ORDER BY m.interval_end
"""

COLUMNS = (
    "demand",
    "price",
    "net_interchange",
    "available",
    "dispatchable",
    "wind_cleared",
    "wind_uigf",
    "solar_cleared",
    "solar_uigf",
)


def business_days(ends: pd.DatetimeIndex, tz: str, calendar: pd.DataFrame) -> np.ndarray:
    """Per interval: its local date (by interval start) is neither a weekend nor a holiday."""
    local_dates = pd.Index((ends - INTERVAL).tz_convert(tz).date)
    cal = calendar.set_index("local_date")
    missing = ~local_dates.isin(cal.index)
    if missing.any():
        raise LookupError(
            f"calendar_days lacks {int(missing.sum())} interval dates, e.g. "
            f"{local_dates[missing][0]}; run `gridcast calendar`"
        )
    off = (cal["is_weekend"].astype(bool) | cal["is_public_holiday"].astype(bool)).to_dict()
    return ~np.array([off[d] for d in local_dates], dtype=bool)


def load_market_series(
    engine: Engine, cfg: AppConfig, region_id: str, last_end: pd.Timestamp | None = None
) -> MarketSeries:
    """The region's complete 5-minute series up to ``last_end`` (UTC interval end)."""
    region = cfg.region(region_id)
    last = last_end if last_end is not None else pd.Timestamp.max.tz_localize("UTC")
    with engine.connect() as conn:
        frame = pd.read_sql(
            text(MARKET_SQL), conn, params={"region_id": region_id, "last_end": last}
        )
        calendar = load_calendar_days(conn, region.holiday_subdiv)
    if frame.empty:
        raise LookupError(f"no market_intervals for {region_id} up to {last}")
    ends = pd.DatetimeIndex(pd.to_datetime(frame["interval_end"], utc=True))
    arrays = {c: frame[c].to_numpy(dtype=np.float64, na_value=np.nan) for c in COLUMNS}
    for c in ("dispatchable", "wind_cleared", "wind_uigf", "solar_cleared", "solar_uigf"):
        n_missing = int(np.isnan(arrays[c]).sum())
        if n_missing:
            raise ValueError(
                f"{c}: {n_missing} intervals without generation_intervals rows; "
                "run `gridcast ingest-generation`"
            )
    return MarketSeries(
        ends=ends,
        business=business_days(ends, region.local_tz, calendar),
        tz=region.local_tz,
        **arrays,
    )


def read_band(
    cfg: AppConfig, qr_batch: str, mask_batch: str, end: pd.Timestamp | None = None
) -> tuple[QuantileBand, np.ndarray]:
    """A stored QR batch's 2.5–97.5% band with the M2 batch's common mask, and its actuals.

    ``end`` keeps issue times before it (the M2 batch is restricted the same way).
    """
    qr_dir = cfg.data_path(cfg.intervals.output_dir) / qr_batch
    keys = [cfg.anomalies.residual.lower_quantile, cfg.anomalies.residual.upper_quantile]
    times, q = read_qr_quantiles(qr_dir, keys)
    a_times, actual = read_wide(qr_dir / "actual.parquet")
    if not a_times.equals(times):
        raise ValueError(f"{QR_RUN_NAME} {qr_batch}: actuals and quantiles differ in issue times")
    batch = read_stored_batch(cfg.data_path(cfg.paths.backtest_dir), mask_batch, "demand")
    keep = np.ones(len(times), dtype=bool) if end is None else np.asarray(times < end)
    batch = batch.restrict(end=end)
    times, actual = times[keep], actual[keep]
    lower, upper = q[keys[0]][keep], q[keys[1]][keep]
    if not times.equals(batch.origin_times):
        raise ValueError(
            f"QR batch {qr_batch} has {len(times)} issue times, M2 batch {mask_batch} has "
            f"{len(batch.origin_times)}: they must match"
        )
    if not np.array_equal(
        actual.astype(np.float32), batch.actual.astype(np.float32), equal_nan=True
    ):
        raise ValueError(f"QR batch {qr_batch}'s actuals differ from M2 batch {mask_batch}'s")
    mask = batch.common_mask()
    missing = int((mask & ~(np.isfinite(lower) & np.isfinite(upper))).sum())
    if missing:
        raise ValueError(f"QR batch {qr_batch} lacks quantiles on {missing} common-mask pairs")
    valid = mask & np.isfinite(lower) & np.isfinite(upper)
    return QuantileBand(pd.DatetimeIndex(times), lower, upper, valid), actual


def check_band_actuals(band: QuantileBand, actual: np.ndarray, series: MarketSeries) -> int:
    """Stored actuals must equal market demand (float32) at every valid pair inside the
    series; returns the number of pairs checked."""
    minutes = epoch_minutes(series.ends)
    issue = epoch_minutes(band.issue_times)
    leads = np.arange(1, band.n_leads + 1)
    idx = (issue[:, None] + 5 * leads[None, :] - minutes[0]) // 5
    inside = band.valid & (idx >= 0) & (idx < len(series))
    stored = actual[inside].astype(np.float32)
    market = series.demand[idx[inside]].astype(np.float32)
    bad = int((stored != market).sum())
    if bad:
        raise ValueError(f"{bad} stored actuals differ from market demand")
    return int(inside.sum())
