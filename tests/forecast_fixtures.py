"""Synthetic SeriesContext builders for forecasting tests (no database)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from functools import cache

import numpy as np
import pandas as pd

from gridcast.features.calendar import build_calendar_days
from gridcast.forecasting.origins import SeriesContext, build_context
from gridcast.timeutils import INTERVAL

MEL = "Australia/Melbourne"


@cache
def calendar() -> pd.DataFrame:
    return build_calendar_days("VIC", MEL, dt.date(2021, 1, 1), dt.date(2027, 12, 31))


def make_ctx(
    start: str = "2024-03-01 00:05",
    days: int = 40,
    values: Callable[[np.ndarray], np.ndarray] | None = None,
) -> SeriesContext:
    """Context with ``days`` of 5-min data; default values = position (easy to reason about)."""
    ends = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=days * 288, freq=INTERVAL)
    pos = np.arange(len(ends), dtype=np.float64)
    y = values(pos) if values is not None else pos
    market = pd.DataFrame(
        {
            "interval_end": ends,
            "total_demand_mw": y,
            "rrp_aud_mwh": y,
            "apc_flag_bits": np.zeros(len(ends), dtype=np.int16),
            "market_suspended_flag": np.zeros(len(ends), dtype=np.int16),
        }
    )
    return build_context(market, "total_demand_mw", MEL, calendar())
