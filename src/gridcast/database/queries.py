"""Read helpers returning pandas frames."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, text

from gridcast.config import WEATHER_VARIABLES

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


def load_weather_sites(engine: Engine, region_id: str) -> pd.DataFrame:
    """Weather sites of a region with their population weights."""
    with engine.connect() as conn:
        return pd.read_sql(
            text(
                """
                SELECT site_id, name, latitude, longitude, population, weight
                FROM weather_sites WHERE region_id = :region_id ORDER BY site_id
                """
            ),
            conn,
            params={"region_id": region_id},
        )


def _weather_times(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    for col in columns:
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def load_weather_forecasts(
    engine: Engine, model: str, first: pd.Timestamp, last: pd.Timestamp
) -> pd.DataFrame:
    """Archived forecasts with ``valid_time`` in ``[first, last]``, every site and offset."""
    with engine.connect() as conn:
        df = pd.read_sql(
            text(
                f"""
                SELECT site_id, offset_days, valid_time, run_init, available_at,
                       {", ".join(WEATHER_VARIABLES)}
                FROM weather_forecasts
                WHERE model = :model AND valid_time BETWEEN :first AND :last
                ORDER BY offset_days, site_id, valid_time
                """
            ),
            conn,
            params={"model": model, "first": first, "last": last},
        )
    df["offset_days"] = df["offset_days"].astype("int16")
    return _weather_times(df, ("valid_time", "run_init", "available_at"))


def load_weather_observations(
    engine: Engine, model: str, first: pd.Timestamp, last: pd.Timestamp
) -> pd.DataFrame:
    """Reanalysis values with ``valid_time`` in ``[first, last]``, every site."""
    with engine.connect() as conn:
        df = pd.read_sql(
            text(
                f"""
                SELECT site_id, valid_time, {", ".join(WEATHER_VARIABLES)}
                FROM weather_observations
                WHERE model = :model AND valid_time BETWEEN :first AND :last
                ORDER BY site_id, valid_time
                """
            ),
            conn,
            params={"model": model, "first": first, "last": last},
        )
    return _weather_times(df, ("valid_time",))
