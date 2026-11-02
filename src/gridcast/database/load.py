"""Bulk loading into PostgreSQL: COPY into a temporary staging table, then upsert.

Re-running a load is idempotent: rows are keyed by ``(region_id, interval_end)`` and
only rows whose values actually changed are rewritten.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Iterator, Sequence
from typing import Any

import pandas as pd
from sqlalchemy import Connection, text

from gridcast.config import AppConfig

MARKET_COLUMNS: tuple[str, ...] = (
    "region_id",
    "interval_start",
    "interval_end",
    "total_demand_mw",
    "rrp_aud_mwh",
    "net_interchange_mw",
    "available_generation_mw",
    "apc_flag",
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


def upsert_market_intervals(conn: Connection, df: pd.DataFrame, ingestion_run_id: int) -> int:
    """COPY ``df`` into a staging table and upsert into ``market_intervals``.

    Returns the number of rows inserted or changed (unchanged rows are skipped).
    """
    if df.empty:
        return 0
    conn.execute(
        text(
            """
            CREATE TEMP TABLE stage_market_intervals
                (LIKE market_intervals INCLUDING DEFAULTS) ON COMMIT DROP
            """
        )
    )
    cols = ", ".join(MARKET_COLUMNS)
    driver_conn = conn.connection.driver_connection
    assert driver_conn is not None
    with (
        driver_conn.cursor() as cur,
        cur.copy(f"COPY stage_market_intervals ({cols}) FROM STDIN") as copy,
    ):
        for row in _rows(df, MARKET_COLUMNS):
            copy.write_row(row)

    set_clause = ",\n".join(f"{c} = EXCLUDED.{c}" for c in _VALUE_COLUMNS)
    changed = " OR ".join(f"t.{c} IS DISTINCT FROM EXCLUDED.{c}" for c in _VALUE_COLUMNS)
    result = conn.execute(
        text(
            f"""
            INSERT INTO market_intervals AS t ({cols}, ingestion_run_id)
            SELECT {cols}, :run_id FROM stage_market_intervals
            ON CONFLICT (region_id, interval_end) DO UPDATE
               SET {set_clause},
                   ingestion_run_id = EXCLUDED.ingestion_run_id,
                   loaded_at = now()
             WHERE {changed}
            """
        ),
        {"run_id": ingestion_run_id},
    )
    conn.execute(text("DROP TABLE IF EXISTS stage_market_intervals"))
    return int(result.rowcount)
