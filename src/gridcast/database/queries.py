"""Read helpers returning pandas frames."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, text

MARKET_SELECT = """
SELECT interval_end, total_demand_mw, rrp_aud_mwh, apc_flag_bits, market_suspended_flag
FROM market_intervals
WHERE region_id = :region_id
ORDER BY interval_end
"""


def load_market_frame(engine: Engine, region_id: str) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(text(MARKET_SELECT), conn, params={"region_id": region_id})
    if df.empty:
        raise LookupError(f"no market_intervals for {region_id}; run `gridcast ingest` first")
    df["interval_end"] = pd.to_datetime(df["interval_end"], utc=True)
    df["apc_flag_bits"] = df["apc_flag_bits"].astype("int16")
    df["market_suspended_flag"] = df["market_suspended_flag"].astype("int16")
    return df
