"""Synthetic, clean market-interval frames for integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gridcast.timeutils import INTERVAL, month_bounds_utc


def clean_month(year: int = 2021, month: int = 10, region: str = "VIC1") -> pd.DataFrame:
    first, last = month_bounds_utc(year, month)
    ends = pd.date_range(first, last, freq=INTERVAL)
    n = len(ends)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "region_id": pd.Series([region] * n, dtype="string"),
            "interval_start": ends - INTERVAL,
            "interval_end": ends,
            "total_demand_mw": 4500 + 800 * np.sin(np.arange(n) * 2 * np.pi / 288),
            "rrp_aud_mwh": rng.normal(60, 20, n),
            "net_interchange_mw": rng.normal(0, 300, n),
            "available_generation_mw": np.full(n, 9000.0),
            "apc_flag_bits": np.zeros(n, dtype=np.int16),
            "market_suspended_flag": np.zeros(n, dtype=np.int16),
            "price_status": pd.Series(["FIRM"] * n, dtype="string"),
        }
    )
