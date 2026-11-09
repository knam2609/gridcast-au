"""Timestamp conventions shared by ingestion, features and evaluation.

* AEMO "NEM time" is AEST: a **fixed** UTC+10 offset all year (no daylight saving).
* Storage and all arithmetic use UTC.
* Local (e.g. Australia/Melbourne, which observes DST) time is derived only for calendar
  features, from ``interval_start``.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

NEM_TZ = dt.timezone(dt.timedelta(hours=10), name="NEM")
NEM_TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M:%S"
INTERVAL = pd.Timedelta(minutes=5)


def parse_nem_timestamps(values: pd.Series) -> pd.Series:
    """Parse AEMO ``YYYY/MM/DD HH:MM:SS`` strings (NEM time) into UTC timestamps.

    Raises ``ValueError`` on any unparseable (or missing) value rather than coercing it.
    """
    if values.isna().any():
        raise ValueError(f"{int(values.isna().sum())} missing timestamps")
    parsed = pd.to_datetime(values, format=NEM_TIMESTAMP_FORMAT, errors="raise")
    return parsed.dt.tz_localize(NEM_TZ).dt.tz_convert("UTC")


def nem_naive_to_utc(ts: dt.datetime) -> pd.Timestamp:
    """Interpret a naive datetime as NEM time and return it in UTC."""
    if ts.tzinfo is not None:
        raise ValueError("expected a naive NEM-time datetime")
    return pd.Timestamp(ts.replace(tzinfo=NEM_TZ)).tz_convert("UTC")


def month_bounds_utc(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """``(first_interval_end, last_interval_end)`` in UTC for an MMSDM archive month.

    A month's archive covers intervals ending ``YYYY-MM-01 00:05`` through
    ``YYYY-(MM+1)-01 00:00`` in NEM time.
    """
    start = pd.Timestamp(year=year, month=month, day=1, tz=NEM_TZ)
    end = start + pd.offsets.MonthBegin(1)
    return (start + INTERVAL).tz_convert("UTC"), end.tz_convert("UTC")


def month_hours_utc(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last whole hour (UTC, inclusive) that hourly data must cover for a month.

    The month's intervals end in ``(start, end]`` (NEM time), so hourly values are needed
    from ``floor_1h`` of the first interval to ``ceil_1h`` of the last: ``[start, end]``.
    NEM time is a whole-hour offset from UTC, so both bounds are whole UTC hours.
    """
    start = pd.Timestamp(year=year, month=month, day=1, tz=NEM_TZ)
    end = start + pd.offsets.MonthBegin(1)
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def to_local(utc_values: pd.Series, tz: str) -> pd.Series:
    """Convert UTC timestamps to a local IANA time zone (DST-aware)."""
    return utc_values.dt.tz_convert(ZoneInfo(tz))


def local_day_n_intervals(local_date: dt.date, tz: str, minutes: int = 5) -> int:
    """Number of ``minutes``-long intervals in a local calendar day (DST-aware).

    Australia/Melbourne: 288 normally, 276 when clocks go forward, 300 when they go back.
    """
    zone = ZoneInfo(tz)
    start = dt.datetime.combine(local_date, dt.time(), tzinfo=zone)
    end = dt.datetime.combine(local_date + dt.timedelta(days=1), dt.time(), tzinfo=zone)
    seconds = (end.astimezone(dt.UTC) - start.astimezone(dt.UTC)).total_seconds()
    return int(seconds // (minutes * 60))
