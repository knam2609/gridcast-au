"""Leakage-safe forecast geometry: origins, horizons and the information set.

Definitions (all positions index a complete, regular 5-minute grid of ``interval_end``):

* An **origin** ``o`` is the issue time of a forecast. Origins fall every
  ``issue_every_minutes`` (30 min) on the UTC grid.
* The **information set** at ``o`` is every interval with ``interval_end <= o - latency``.
  With the default ``latency_steps = 0``, the interval ending exactly at ``o`` is known.
* The **target** for horizon ``h`` (1…H) is the interval ending at ``t = o + h * 5 min``.
* Only *calendar* attributes of ``t`` (known in advance) may be used at target time;
  every observed quantity must come from the information set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridcast.config import AppConfig
from gridcast.features.calendar import interval_calendar_features
from gridcast.timeutils import INTERVAL, NEM_TZ


@dataclass(frozen=True, slots=True)
class ForecastSetup:
    issue_every_steps: int = 6
    horizon_steps: int = 288
    latency_steps: int = 0

    @classmethod
    def from_config(cls, cfg: AppConfig) -> ForecastSetup:
        step = cfg.nem.dispatch_interval_minutes
        if cfg.forecast.issue_every_minutes % step:
            raise ValueError("issue_every_minutes must be a multiple of the dispatch interval")
        return cls(
            issue_every_steps=cfg.forecast.issue_every_minutes // step,
            horizon_steps=cfg.forecast.horizon_steps,
            latency_steps=cfg.forecast.availability_latency_steps,
        )

    @property
    def horizons(self) -> np.ndarray:
        return np.arange(1, self.horizon_steps + 1)


@dataclass(slots=True)
class SeriesContext:
    """A target series on a complete 5-min grid plus target-time calendar arrays.

    Arrays are aligned by position with ``times`` (``interval_end``, UTC).
    """

    times: pd.DatetimeIndex
    tz: str  # local IANA zone used for calendar attributes
    y: np.ndarray  # float64, NaN where missing
    local_day: np.ndarray  # int64 local-date ordinal of interval_start
    day_type: np.ndarray  # int8: 0 = working day, 1 = weekend or public holiday
    slot: np.ndarray  # int16: local minute-of-day // 5 of interval_start (0..287)
    frame: pd.DataFrame  # full aligned frame (observed columns + calendar features)

    def __len__(self) -> int:
        return len(self.times)

    def with_y(self, y: np.ndarray) -> SeriesContext:
        """Copy with a different target array (used by leakage tests)."""
        return SeriesContext(
            self.times, self.tz, y, self.local_day, self.day_type, self.slot, self.frame
        )


def regularise(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex an ``interval_end``-indexed frame onto a complete 5-minute grid.

    Missing intervals become NaN (and False for boolean flags) so that positional
    arithmetic (``t - 288`` = same time yesterday) is always valid.
    """
    if df.index.has_duplicates:
        raise ValueError("duplicate interval_end values")
    grid = pd.date_range(df.index.min(), df.index.max(), freq=INTERVAL, name="interval_end")
    out = df.reindex(grid)
    for col in out.columns:
        if df[col].dtype == bool:
            out[col] = out[col].astype("boolean").fillna(False).astype(bool)
    return out


def build_context(
    market: pd.DataFrame, target_column: str, tz: str, calendar_days: pd.DataFrame
) -> SeriesContext:
    """Build a :class:`SeriesContext` from ``market_intervals`` rows (any order)."""
    frame = regularise(market.set_index("interval_end").sort_index())
    times = pd.DatetimeIndex(frame.index)
    starts = pd.Series(times - INTERVAL, index=times)
    cal = interval_calendar_features(starts, tz, calendar_days)
    frame = frame.join(cal)
    local_day = np.array([d.toordinal() for d in cal["local_date"]], dtype=np.int64)
    return SeriesContext(
        times=times,
        tz=tz,
        y=frame[target_column].to_numpy(dtype=np.float64, na_value=np.nan),
        local_day=local_day,
        day_type=(cal["day_type"] == "non_working").to_numpy(dtype=np.int8),
        slot=cal["slot_of_day"].to_numpy(dtype=np.int16),
        frame=frame,
    )


def nem_midnight_utc(day: pd.Timestamp | str) -> pd.Timestamp:
    """Midnight NEM time (UTC+10) of a calendar date, in UTC."""
    ts = pd.Timestamp(day)
    if ts.tzinfo is not None:
        raise ValueError("expected a naive date")
    return ts.normalize().tz_localize(NEM_TZ).tz_convert("UTC")


def origin_positions(
    ctx: SeriesContext,
    setup: ForecastSetup,
    start: pd.Timestamp,
    end: pd.Timestamp | None = None,
) -> np.ndarray:
    """Positions of every origin in ``[start, end]`` whose full horizon lies inside the data.

    Origins sit on the issue grid (multiples of ``issue_every_steps`` x 5 min since the
    Unix epoch, i.e. :00 and :30 for a 30-minute cadence).
    """
    # Unit-agnostic (pandas may store datetimes in ns or us): whole steps since the epoch.
    epoch_steps = ((ctx.times - pd.Timestamp(0, tz="UTC")) // INTERVAL).to_numpy(dtype=np.int64)
    on_grid = epoch_steps % setup.issue_every_steps == 0
    positions = np.arange(len(ctx))
    ok = on_grid & (ctx.times >= start) & (positions + setup.horizon_steps < len(ctx))
    if end is not None:
        ok &= ctx.times <= end
    return positions[ok]


def target_positions(origins: np.ndarray, setup: ForecastSetup) -> np.ndarray:
    """``(n_origins, H)`` matrix of target positions ``o + h``."""
    return origins[:, None] + setup.horizons[None, :]
