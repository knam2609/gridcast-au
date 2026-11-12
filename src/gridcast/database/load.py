"""Bulk loading into PostgreSQL: COPY into a temporary staging table, then upsert.

Re-running a load is idempotent: rows are keyed (market intervals by
``(region_id, interval_end)``, weather by model, site, offset and valid hour) and only rows
whose values actually changed are rewritten.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Iterator, Sequence
from typing import Any

import pandas as pd
from sqlalchemy import Connection, text

from gridcast.config import WEATHER_VARIABLES, AppConfig

MARKET_COLUMNS: tuple[str, ...] = (
    "region_id",
    "interval_start",
    "interval_end",
    "total_demand_mw",
    "rrp_aud_mwh",
    "net_interchange_mw",
    "available_generation_mw",
    "apc_flag_bits",
    "market_suspended_flag",
    "price_status",
)
_VALUE_COLUMNS = MARKET_COLUMNS[3:]


def ensure_regions(conn: Connection, cfg: AppConfig) -> None:
    """Upsert configured regions (adding a NEM region is a config change, not a migration)."""
    for region_id, region in cfg.regions.items():
        conn.execute(
            text(
                """
                INSERT INTO regions (region_id, name, holiday_jurisdiction, local_tz)
                VALUES (:region_id, :name, :jurisdiction, :tz)
                ON CONFLICT (region_id) DO UPDATE
                   SET name = EXCLUDED.name,
                       holiday_jurisdiction = EXCLUDED.holiday_jurisdiction,
                       local_tz = EXCLUDED.local_tz
                """
            ),
            {
                "region_id": region_id,
                "name": region.name,
                "jurisdiction": region.holiday_subdiv,
                "tz": region.local_tz,
            },
        )


def start_ingestion_run(conn: Connection, source: str, dataset: str, period_month: dt.date) -> int:
    run_id = conn.execute(
        text(
            """
            INSERT INTO ingestion_runs (source, dataset, period_month)
            VALUES (:source, :dataset, :period_month)
            RETURNING run_id
            """
        ),
        {"source": source, "dataset": dataset, "period_month": period_month},
    ).scalar_one()
    return int(run_id)


def finish_ingestion_run(
    conn: Connection,
    run_id: int,
    *,
    status: str,
    files: Sequence[dict[str, Any]] = (),
    rows_parsed: int | None = None,
    duplicates_dropped: int | None = None,
    rows_loaded: int | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        text(
            """
            UPDATE ingestion_runs
               SET status = :status,
                   files = CAST(:files AS jsonb),
                   rows_parsed = :rows_parsed,
                   duplicates_dropped = :duplicates_dropped,
                   rows_loaded = :rows_loaded,
                   error = :error,
                   finished_at = now()
             WHERE run_id = :run_id
            """
        ),
        {
            "run_id": run_id,
            "status": status,
            "files": json.dumps(list(files)),
            "rows_parsed": rows_parsed,
            "duplicates_dropped": duplicates_dropped,
            "rows_loaded": rows_loaded,
            "error": error,
        },
    )


def _py(value: Any) -> Any:
    """Convert pandas/numpy scalars to plain Python values psycopg can COPY."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    return value


def _rows(df: pd.DataFrame, columns: Sequence[str]) -> Iterator[tuple[Any, ...]]:
    for record in df[list(columns)].itertuples(index=False, name=None):
        yield tuple(_py(v) for v in record)


def _copy_upsert(
    conn: Connection,
    table: str,
    df: pd.DataFrame,
    columns: Sequence[str],
    key: Sequence[str],
    ingestion_run_id: int,
) -> int:
    """COPY ``df[columns]`` into a temporary staging table, then upsert into ``table``.

    Rows are matched on ``key``; only rows whose values changed are rewritten. ``table`` and
    the column names are module constants, never user input. Returns rows inserted/changed.
    """
    if df.empty:
        return 0
    stage = f"stage_{table}"
    conn.execute(
        text(f"CREATE TEMP TABLE {stage} (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP")
    )
    cols = ", ".join(columns)
    driver_conn = conn.connection.driver_connection
    assert driver_conn is not None
    with (
        driver_conn.cursor() as cur,
        cur.copy(f"COPY {stage} ({cols}) FROM STDIN") as copy,
    ):
        for row in _rows(df, columns):
            copy.write_row(row)

    values = [c for c in columns if c not in key]
    set_clause = ",\n".join(f"{c} = EXCLUDED.{c}" for c in values)
    changed = " OR ".join(f"t.{c} IS DISTINCT FROM EXCLUDED.{c}" for c in values)
    result = conn.execute(
        text(
            f"""
            INSERT INTO {table} AS t ({cols}, ingestion_run_id)
            SELECT {cols}, :run_id FROM {stage}
            ON CONFLICT ({", ".join(key)}) DO UPDATE
               SET {set_clause},
                   ingestion_run_id = EXCLUDED.ingestion_run_id,
                   loaded_at = now()
             WHERE {changed}
            """
        ),
        {"run_id": ingestion_run_id},
    )
    conn.execute(text(f"DROP TABLE IF EXISTS {stage}"))
    return int(result.rowcount)


def upsert_market_intervals(conn: Connection, df: pd.DataFrame, ingestion_run_id: int) -> int:
    """COPY ``df`` into a staging table and upsert into ``market_intervals``.

    Returns the number of rows inserted or changed (unchanged rows are skipped).
    """
    return _copy_upsert(
        conn,
        "market_intervals",
        df,
        MARKET_COLUMNS,
        ("region_id", "interval_end"),
        ingestion_run_id,
    )


WEATHER_OBSERVATION_COLUMNS: tuple[str, ...] = (
    "model",
    "site_id",
    "valid_time",
    *WEATHER_VARIABLES,
)
WEATHER_FORECAST_COLUMNS: tuple[str, ...] = (
    "model",
    "site_id",
    "offset_days",
    "valid_time",
    "run_init",
    "available_at",
    *WEATHER_VARIABLES,
)


def ensure_weather_sites(conn: Connection, cfg: AppConfig) -> None:
    """Upsert configured weather sites with their population weights within each region.

    Sites removed from the config are kept (data references them), so the region's weights
    would no longer sum to 1; the ``weather_site_weights`` data-quality check reports that.
    """
    weather = cfg.weather
    for region_id in sorted({s.region_id for s in weather.sites.values()}):
        weights = weather.site_weights(region_id)
        for site_id, site in weather.sites_for(region_id).items():
            conn.execute(
                text(
                    """
                    INSERT INTO weather_sites (site_id, region_id, name, latitude, longitude,
                                               population, weight, population_source)
                    VALUES (:site_id, :region_id, :name, :lat, :lon, :pop, :weight, :source)
                    ON CONFLICT (site_id) DO UPDATE
                       SET region_id = EXCLUDED.region_id,
                           name = EXCLUDED.name,
                           latitude = EXCLUDED.latitude,
                           longitude = EXCLUDED.longitude,
                           population = EXCLUDED.population,
                           weight = EXCLUDED.weight,
                           population_source = EXCLUDED.population_source,
                           updated_at = now()
                    """
                ),
                {
                    "site_id": site_id,
                    "region_id": region_id,
                    "name": site.name,
                    "lat": site.latitude,
                    "lon": site.longitude,
                    "pop": site.population,
                    "weight": weights[site_id],
                    "source": weather.population.source,
                },
            )


def upsert_weather_observations(conn: Connection, df: pd.DataFrame, ingestion_run_id: int) -> int:
    return _copy_upsert(
        conn,
        "weather_observations",
        df,
        WEATHER_OBSERVATION_COLUMNS,
        ("model", "site_id", "valid_time"),
        ingestion_run_id,
    )


def upsert_weather_forecasts(conn: Connection, df: pd.DataFrame, ingestion_run_id: int) -> int:
    return _copy_upsert(
        conn,
        "weather_forecasts",
        df,
        WEATHER_FORECAST_COLUMNS,
        ("model", "site_id", "offset_days", "valid_time"),
        ingestion_run_id,
    )


AEMO_FORECAST_COLUMNS: tuple[str, ...] = (
    "source",
    "region_id",
    "run_time",
    "target_end",
    "resolution_minutes",
    "available_at",
    "total_demand_mw",
)


def upsert_aemo_forecasts(conn: Connection, df: pd.DataFrame, ingestion_run_id: int) -> int:
    return _copy_upsert(
        conn,
        "aemo_demand_forecasts",
        df,
        AEMO_FORECAST_COLUMNS,
        ("source", "region_id", "run_time", "target_end"),
        ingestion_run_id,
    )


AEMO_POE_COLUMNS: tuple[str, ...] = (
    "region_id",
    "run_time",
    "target_end",
    "available_at",
    "demand10_mw",
    "demand50_mw",
    "demand90_mw",
)


def upsert_aemo_poe(conn: Connection, df: pd.DataFrame, ingestion_run_id: int) -> int:
    return _copy_upsert(
        conn,
        "aemo_demand_poe",
        df,
        AEMO_POE_COLUMNS,
        ("region_id", "run_time", "target_end"),
        ingestion_run_id,
    )
