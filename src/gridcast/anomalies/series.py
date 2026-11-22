"""The detectors' inputs on one complete 5-minute grid, and its local-time layout.

* :class:`MarketSeries`: Victorian demand, price and supply per interval (``market_intervals``
  and ``generation_intervals``), plus the day type of each interval's local day.
* :class:`QuantileBand`: M3's stored raw quantile LightGBM C band (rearranged), per issue
  time and lead, with M2's common mask (the pairs with C weather).
* :class:`Layout`: each interval's local day and local half-hour slot (Australia/Melbourne,
  by interval start, as M2's calendar features), and its NEM month.

Times are interval ends in UTC, as in ``market_intervals``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridcast.timeutils import INTERVAL, NEM_TZ

MARKET_ARRAYS = (
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
STEP_NS = INTERVAL.value
INTERVALS_PER_DAY = 288  # NEM days have no daylight saving
ISSUE_STEP_MINUTES = 30


def epoch_ns(times: pd.DatetimeIndex) -> np.ndarray:
    """Nanoseconds since the epoch, whatever the index's unit."""
    return np.asarray(times.as_unit("ns").asi8, dtype=np.int64)


def epoch_minutes(times: pd.DatetimeIndex) -> np.ndarray:
    return epoch_ns(times) // (60 * 10**9)


def check_grid(ends: pd.DatetimeIndex) -> None:
    """``ends`` must be UTC, on the 5-minute grid and complete (no gaps, no repeats)."""
    if ends.tz is None or str(ends.tz) != "UTC":
        raise ValueError("interval ends must be UTC")
    if len(ends) == 0:
        raise ValueError("empty series")
    ns = epoch_ns(ends)
    if np.any(ns % STEP_NS):
        raise ValueError("interval ends must lie on the 5-minute grid")
    if len(ns) > 1 and np.any(np.diff(ns) != STEP_NS):
        raise ValueError("the 5-minute grid must be complete and ordered")


@dataclass(frozen=True, slots=True)
class MarketSeries:
    """One region's market and supply values on a complete 5-minute grid."""

    ends: pd.DatetimeIndex
    demand: np.ndarray  # MW, TOTALDEMAND
    price: np.ndarray  # $/MWh, RRP
    net_interchange: np.ndarray  # MW (positive = export, as AEMO)
    available: np.ndarray  # MW, AVAILABLEGENERATION
    dispatchable: np.ndarray  # MW, DISPATCHABLEGENERATION
    wind_cleared: np.ndarray  # MW, semi-scheduled
    wind_uigf: np.ndarray  # MW, AEMO's unconstrained intermittent generation forecast
    solar_cleared: np.ndarray
    solar_uigf: np.ndarray
    business: np.ndarray  # bool: the interval's local day is a business day
    tz: str  # local time zone (profiles, day types)

    def __post_init__(self) -> None:
        check_grid(self.ends)
        n = len(self.ends)
        for name in (*MARKET_ARRAYS, "business"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have one value per interval")

    def __len__(self) -> int:
        return len(self.ends)

    def head(self, n: int) -> MarketSeries:
        """The first ``n`` intervals (data truncated after interval ``n − 1``)."""
        arrays = {k: getattr(self, k)[:n] for k in (*MARKET_ARRAYS, "business")}
        return MarketSeries(ends=self.ends[:n], tz=self.tz, **arrays)

    def with_values(self, **arrays: np.ndarray) -> MarketSeries:
        """A copy with some arrays replaced."""
        unknown = set(arrays) - {*MARKET_ARRAYS, "business"}
        if unknown:
            raise ValueError(f"unknown arrays {sorted(unknown)}")
        current = {k: getattr(self, k) for k in (*MARKET_ARRAYS, "business")}
        return MarketSeries(ends=self.ends, tz=self.tz, **{**current, **arrays})


@dataclass(frozen=True, slots=True)
class QuantileBand:
    """Stored lower/upper quantiles per issue time (30 minutes apart) and lead."""

    issue_times: pd.DatetimeIndex  # UTC
    lower: np.ndarray  # (m, H)
    upper: np.ndarray  # (m, H)
    valid: np.ndarray  # (m, H) bool: M2's scored pairs with finite quantiles

    def __post_init__(self) -> None:
        times = self.issue_times
        if times.tz is None or str(times.tz) != "UTC":
            raise ValueError("issue times must be UTC")
        if len(times) > 1 and not times.is_monotonic_increasing:
            raise ValueError("issue times must be increasing")
        if np.any(epoch_ns(times) % (ISSUE_STEP_MINUTES * 60 * 10**9)):
            raise ValueError("issue times must lie on the 30-minute grid")
        shape = (len(times), self.lower.shape[1] if self.lower.ndim == 2 else -1)
        for name in ("lower", "upper", "valid"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must be (issue times, leads)")
        if np.any(self.valid & ~(np.isfinite(self.lower) & np.isfinite(self.upper))):
            raise ValueError("a valid pair has a missing quantile")

    @property
    def n_leads(self) -> int:
        return int(self.lower.shape[1])

    def issued_until(self, last: pd.Timestamp) -> QuantileBand:
        """Issue times up to ``last`` (forecasts known by then)."""
        keep = np.asarray(self.issue_times <= last)
        return QuantileBand(
            self.issue_times[keep], self.lower[keep], self.upper[keep], self.valid[keep]
        )


@dataclass(frozen=True, slots=True)
class Layout:
    """Local day, local slot and NEM month of every interval (all by interval start)."""

    local_day: np.ndarray  # (n,) int64: days since the first interval's local date
    slot: np.ndarray  # (n,) int64: local time of day // slot_minutes
    n_days: int
    n_slots: int
    day_business: np.ndarray  # (n_days,) bool; days without an interval count as business
    nem_month: np.ndarray  # (n,) int64: 12 * year + month − 1 of the NEM-time start


def layout(series: MarketSeries, slot_minutes: int) -> Layout:
    starts = series.ends - INTERVAL
    local = starts.tz_convert(series.tz)
    dates = epoch_ns(local.normalize().tz_localize(None)) // (86_400 * 10**9)
    local_day = (dates - dates[0]).astype(np.int64)
    minute = np.asarray(local.hour * 60 + local.minute, dtype=np.int64)
    n_days = int(local_day[-1]) + 1
    day_business = np.ones(n_days, dtype=bool)
    day_business[local_day] = series.business
    nem = starts.tz_convert(NEM_TZ)
    return Layout(
        local_day=local_day,
        slot=minute // slot_minutes,
        n_days=n_days,
        n_slots=(24 * 60) // slot_minutes,
        day_business=day_business,
        nem_month=np.asarray(nem.year * 12 + nem.month - 1, dtype=np.int64),
    )


def nem_day_ordinals(ends: pd.DatetimeIndex) -> np.ndarray:
    """NEM-time date (as ``date.toordinal()``) of each interval's start."""
    starts = (ends - INTERVAL).tz_convert(NEM_TZ)
    days = epoch_ns(starts.normalize().tz_localize(None)) // (86_400 * 10**9)
    return np.asarray(days, dtype=np.int64) + pd.Timestamp("1970-01-01").toordinal()
