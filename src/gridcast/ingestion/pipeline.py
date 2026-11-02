"""ETL orchestration: MMSDM month → parse → standardise → validate → PostgreSQL."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera.errors as pe
from sqlalchemy import Engine

from gridcast.config import AppConfig
from gridcast.database.load import (
    ensure_regions,
    finish_ingestion_run,
    start_ingestion_run,
    upsert_market_intervals,
)
from gridcast.ingestion.dispatch import (
    DISPATCHPRICE,
    DISPATCHREGIONSUM,
    TableExtract,
    build_market_intervals,
    extract_table,
)
from gridcast.ingestion.nemweb import DownloadedFile, NemwebClient
from gridcast.logs import get_logger
from gridcast.validation.schemas import summarise_schema_errors, validate_market_intervals

log = get_logger(__name__)

SOURCE = "mmsdm"
DATASET = "market_intervals"


@dataclass(frozen=True, slots=True)
class MonthResult:
    period: dt.date
    status: str
    rows_parsed: int = 0
    duplicates_dropped: int = 0
    rows_loaded: int = 0
    error: str | None = None


def parse_month_arg(value: str) -> dt.date:
    """Parse ``YYYY-MM`` into the first day of that month."""
    try:
        return dt.datetime.strptime(value, "%Y-%m").date()  # noqa: DTZ007 - date only
    except ValueError as exc:
        raise ValueError(f"expected YYYY-MM, got {value!r}") from exc


def iter_months(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    """Inclusive month range."""
    if end < start:
        raise ValueError(f"end month {end:%Y-%m} is before start month {start:%Y-%m}")
    current = start.replace(day=1)
    while current <= end:
        yield current
        current = (current.replace(day=28) + dt.timedelta(days=4)).replace(day=1)


def _file_records(downloads: Sequence[DownloadedFile]) -> list[dict[str, object]]:
    return [
        {
            "table": d.file.table,
            "file": d.file.filename,
            "sha256": d.sha256,
            "bytes": d.n_bytes,
            "from_cache": d.from_cache,
        }
        for d in downloads
    ]


def transform_month(
    demand_paths: Sequence[Path], price_paths: Sequence[Path], regions: Sequence[str]
) -> tuple[pd.DataFrame, TableExtract, TableExtract]:
    """Pure transformation step (no I/O beyond reading the given files)."""
    demand = extract_table(demand_paths, DISPATCHREGIONSUM, regions)
    price = extract_table(price_paths, DISPATCHPRICE, regions)
    merged = build_market_intervals(demand.frame, price.frame)
    try:
        validated = validate_market_intervals(merged, regions)
    except pe.SchemaErrors as exc:
        raise ValueError(f"schema validation failed: {summarise_schema_errors(exc)}") from exc
    return validated, demand, price


def ingest_month(
    engine: Engine,
    client: NemwebClient,
    cfg: AppConfig,
    period: dt.date,
    regions: Sequence[str],
) -> MonthResult:
    """Ingest one MMSDM month. Failures are recorded in ``ingestion_runs``, not raised."""
    with engine.begin() as conn:
        ensure_regions(conn, cfg)
        run_id = start_ingestion_run(conn, SOURCE, DATASET, period)
    bound = log.bind(period=f"{period:%Y-%m}", run_id=run_id)

    downloads: list[DownloadedFile] = []
    try:
        demand_files = client.fetch_month(DISPATCHREGIONSUM.name, period.year, period.month)
        price_files = client.fetch_month(DISPATCHPRICE.name, period.year, period.month)
        downloads = [*demand_files, *price_files]
        frame, demand, price = transform_month(
            [d.path for d in demand_files], [d.path for d in price_files], regions
        )
        interim = cfg.resolve(cfg.paths.interim_dir) / DATASET / f"{period:%Y%m}.parquet"
        interim.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(interim, index=False)

        rows_parsed = demand.rows_parsed + price.rows_parsed
        dupes = demand.duplicates_dropped + price.duplicates_dropped
        with engine.begin() as conn:
            loaded = upsert_market_intervals(conn, frame, run_id)
            finish_ingestion_run(
                conn,
                run_id,
                status="success",
                files=_file_records(downloads),
                rows_parsed=rows_parsed,
                duplicates_dropped=dupes,
                rows_loaded=loaded,
            )
        bound.info(
            "ingest.month_ok",
            rows=len(frame),
            rows_loaded=loaded,
            duplicates_dropped=dupes,
            intervention_rows_dropped=demand.intervention_rows_dropped
            + price.intervention_rows_dropped,
            table_versions={"demand": sorted(demand.versions), "price": sorted(price.versions)},
        )
        return MonthResult(period, "success", rows_parsed, dupes, loaded)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        bound.error("ingest.month_failed", error=message)
        with engine.begin() as conn:
            finish_ingestion_run(
                conn, run_id, status="failed", files=_file_records(downloads), error=message[:4000]
            )
        return MonthResult(period, "failed", error=message)


def ingest_range(
    engine: Engine,
    cfg: AppConfig,
    start: dt.date,
    end: dt.date,
    regions: Sequence[str],
    client: NemwebClient | None = None,
) -> list[MonthResult]:
    """Ingest every month in ``[start, end]``; continues past failed months."""
    for region in regions:
        cfg.region(region)  # fail fast on unknown regions
    own_client = client is None
    nemweb = client or NemwebClient(cfg.sources.mmsdm, cfg.resolve(cfg.paths.raw_dir))
    try:
        return [ingest_month(engine, nemweb, cfg, m, regions) for m in iter_months(start, end)]
    finally:
        if own_client:
            nemweb.close()
