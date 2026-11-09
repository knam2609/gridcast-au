"""AEMO's own demand forecasts (M2 benchmark only, never a model input): every run of
30-minute pre-dispatch (``PREDISPATCHREGIONSUM``) and 5-minute pre-dispatch
(``P5MIN_REGIONSOLUTION``) from the MMSDM "all runs" archives → ``aemo_demand_forecasts``.

Loader rules (``docs/m2_data_spike.md`` §4-6, ``docs/plans/m2_plan.md`` §2.6):

* columns are mapped by name; only the table versions seen in the benchmark window are
  accepted (columns were appended in 2025-04 without a version bump), anything else fails;
* only ``INTERVENTION = 0`` rows are kept, and a duplicate (region, run, target) fails;
* ``available_at`` is ``LASTCHANGED``.

Run times (UTC after conversion from NEM time):

* pre-dispatch: ``PREDISPATCHSEQNO`` = ``YYYYMMDDPP``, trading day plus run number; the
  nominal run time is 04:00 NEM time on that day + (PP − 1) × 30 min. ``DATETIME`` is the
  period end, and each run's first period ends 30 min after its run time;
* 5-minute pre-dispatch: ``RUN_DATETIME``; its intervals end at the run time (interval 0,
  equal to the dispatch demand) and 5 … 55 min after it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from gridcast.config import AppConfig
from gridcast.database.load import (
    ensure_regions,
    finish_ingestion_run,
    start_ingestion_run,
    upsert_aemo_forecasts,
)
from gridcast.ingestion.dispatch import _to_float
from gridcast.ingestion.mms_csv import MmsFormatError, ReportKey, read_mms_report
from gridcast.ingestion.nemweb import DownloadedFile, NemwebClient
from gridcast.ingestion.pipeline import SOURCE, MonthResult, iter_months
from gridcast.logs import get_logger
from gridcast.timeutils import NEM_TZ, parse_nem_timestamps

log = get_logger(__name__)

TRADING_DAY_START = dt.timedelta(hours=4)  # NEM trading day starts 04:00 NEM time


@dataclass(frozen=True, slots=True)
class ForecastTable:
    """One AEMO forecast archive: where it lives and how its rows map to our table."""

    source: str  # value of aemo_demand_forecasts.source
    table: str  # MMSDM table name in the archive file name
    folder: str  # month sub-folder holding the "all runs" archive
    key: ReportKey
    versions: frozenset[str]  # accepted MMS table versions
    columns: tuple[str, ...]
    resolution_minutes: int

    @property
    def dataset(self) -> str:
        return f"aemo_{self.source}"


PREDISPATCH = ForecastTable(
    source="predispatch",
    table="PREDISPATCHREGIONSUM",
    folder="PREDISP_ALL_DATA",
    key=ReportKey("PREDISPATCH", "REGION_SOLUTION"),
    versions=frozenset({"8", "9"}),
    columns=(
        "PREDISPATCHSEQNO",
        "RUNNO",
        "REGIONID",
        "PERIODID",
        "INTERVENTION",
        "TOTALDEMAND",
        "LASTCHANGED",
        "DATETIME",
    ),
    resolution_minutes=30,
)

P5MIN = ForecastTable(
    source="p5min",
    table="P5MIN_REGIONSOLUTION",
    folder="P5MIN_ALL_DATA",
    key=ReportKey("P5MIN", "REGIONSOLUTION"),
    versions=frozenset({"9", "10"}),
    columns=(
        "RUN_DATETIME",
        "INTERVAL_DATETIME",
        "REGIONID",
        "INTERVENTION",
        "TOTALDEMAND",
        "LASTCHANGED",
    ),
    resolution_minutes=5,
)

FORECAST_TABLES: tuple[ForecastTable, ...] = (PREDISPATCH, P5MIN)
KEY = ["source", "region_id", "run_time", "target_end"]
OUTPUT_COLUMNS = [*KEY, "resolution_minutes", "available_at", "total_demand_mw"]


@dataclass(slots=True)
class ForecastExtract:
    table: ForecastTable
    frame: pd.DataFrame
    rows_parsed: int
    intervention_rows_dropped: int
    versions: set[str]


def predispatch_run_time(seqno: pd.Series) -> pd.Series:
    """Nominal run time (UTC) from ``PREDISPATCHSEQNO`` = ``YYYYMMDDPP``."""
    s = seqno.str.strip()
    bad = ~s.str.fullmatch(r"\d{10}").fillna(False)
    if bad.any():
        raise MmsFormatError(f"PREDISPATCHSEQNO: malformed values, e.g. {s[bad].head(3).tolist()}")
    run_no = s.str[8:].astype("int64")
    if not run_no.between(1, 48).all():
        raise MmsFormatError(f"PREDISPATCHSEQNO: run numbers outside 1..48: {sorted(set(run_no))}")
    day = pd.to_datetime(s.str[:8], format="%Y%m%d", errors="raise")
    nem = day + TRADING_DAY_START + pd.to_timedelta((run_no - 1) * 30, unit="min")
    return nem.dt.tz_localize(NEM_TZ).dt.tz_convert("UTC")


def _offsets(frame: pd.DataFrame) -> pd.Series:
    return frame["target_end"] - frame["run_time"]


def standardise(raw: pd.DataFrame, table: ForecastTable) -> tuple[pd.DataFrame, int]:
    """Keep the pricing run, derive run/target/availability times (UTC), validate structure.

    Returns ``(frame, intervention_rows_dropped)``. Raises :class:`MmsFormatError` on a
    duplicate key or a run whose periods do not sit where the archive's layout says.
    """
    keep = raw["INTERVENTION"].str.strip().eq("0")
    dropped = int((~keep).sum())
    df = raw.loc[keep]

    if table is PREDISPATCH:
        run_time = predispatch_run_time(df["PREDISPATCHSEQNO"])
        target_end = parse_nem_timestamps(df["DATETIME"])
    else:
        run_time = parse_nem_timestamps(df["RUN_DATETIME"])
        target_end = parse_nem_timestamps(df["INTERVAL_DATETIME"])
    out = pd.DataFrame(
        {
            "source": table.source,
            "region_id": df["REGIONID"].str.strip().astype("string"),
            "run_time": run_time,
            "target_end": target_end,
            "resolution_minutes": table.resolution_minutes,
            "available_at": parse_nem_timestamps(df["LASTCHANGED"]),
            "total_demand_mw": _to_float(df["TOTALDEMAND"], f"{table.table}.TOTALDEMAND"),
        }
    ).reset_index(drop=True)
    if out["total_demand_mw"].isna().any():
        raise MmsFormatError(f"{table.table}: {int(out.total_demand_mw.isna().sum())} blank demand")

    step = table.resolution_minutes
    seconds = _offsets(out).dt.total_seconds()
    offset = (seconds // 60).astype("int64")  # minutes from the run time to the period end
    on_step = (seconds % 60 == 0) & (offset % step == 0)
    if table is PREDISPATCH:
        # Periods end 30, 60, ... min after the run time; the first one ends 30 min after.
        first = offset.groupby([out["region_id"], out["run_time"]]).transform("min")
        off_grid = ~on_step | (offset <= 0) | (first != step)
    else:
        # Interval 0 ends at the run time, then 5 ... 55 min after it.
        off_grid = ~on_step | (offset < 0) | (offset > 11 * step)
    if off_grid.any():
        sample = out.loc[off_grid, ["run_time", "target_end"]].head(3).to_dict("records")
        raise MmsFormatError(
            f"{table.table}: {int(off_grid.sum())} periods off the run grid: {sample}"
        )

    dupes = out.duplicated(subset=KEY, keep=False)
    if dupes.any():
        sample = out.loc[dupes, KEY].head(4).to_dict("records")
        raise MmsFormatError(f"{table.table}: {int(dupes.sum())} rows share a key, e.g. {sample}")
    return out.sort_values(KEY).reset_index(drop=True)[OUTPUT_COLUMNS], dropped


def extract_forecasts(
    paths: Sequence[Path], table: ForecastTable, regions: Collection[str]
) -> ForecastExtract:
    """Parse, check the table version of, and standardise all parts of one archive month."""
    frames: list[pd.DataFrame] = []
    versions: set[str] = set()
    for path in paths:
        frame, stats = read_mms_report(path, table.key, table.columns, region_filter=regions)
        frames.append(frame)
        versions |= stats.versions
    unknown = versions - table.versions
    if unknown:
        raise MmsFormatError(
            f"{table.table}: unsupported version(s) {sorted(unknown)} "
            f"(accepted {sorted(table.versions)}); check the column mapping first"
        )
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=table.columns)
    clean, dropped = standardise(raw, table)
    return ForecastExtract(table, clean, len(raw), dropped, versions)


def ingest_forecast_month(
    engine: Engine,
    client: NemwebClient,
    cfg: AppConfig,
    table: ForecastTable,
    period: dt.date,
    regions: Sequence[str],
) -> MonthResult:
    """Ingest one archive month of one forecast table; failures are recorded, not raised."""
    with engine.begin() as conn:
        ensure_regions(conn, cfg)
        run_id = start_ingestion_run(conn, SOURCE, table.dataset, period)
    bound = log.bind(table=table.table, period=f"{period:%Y-%m}", run_id=run_id)
    downloads: list[DownloadedFile] = []
    try:
        downloads = client.fetch_month(
            table.table, period.year, period.month, folder=table.folder, all_runs=True
        )
        extract = extract_forecasts([d.path for d in downloads], table, regions)
        interim = cfg.data_path(cfg.paths.interim_dir) / table.dataset / f"{period:%Y%m}.parquet"
        interim.parent.mkdir(parents=True, exist_ok=True)
        extract.frame.to_parquet(interim, index=False)
        files = [
            {
                "table": d.file.table,
                "file": d.file.filename,
                "sha256": d.sha256,
                "bytes": d.n_bytes,
                "from_cache": d.from_cache,
            }
            for d in downloads
        ]
        with engine.begin() as conn:
            loaded = upsert_aemo_forecasts(conn, extract.frame, run_id)
            finish_ingestion_run(
                conn,
                run_id,
                status="success",
                files=files,
                rows_parsed=extract.rows_parsed,
                duplicates_dropped=0,
                rows_loaded=loaded,
            )
        bound.info(
            "ingest_aemo.month_ok",
            rows=len(extract.frame),
            rows_loaded=loaded,
            runs=int(extract.frame.groupby(["region_id", "run_time"]).ngroups),
            intervention_rows_dropped=extract.intervention_rows_dropped,
            versions=sorted(extract.versions),
        )
        return MonthResult(period, "success", extract.rows_parsed, 0, loaded)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        bound.error("ingest_aemo.month_failed", error=message)
        with engine.begin() as conn:
            finish_ingestion_run(conn, run_id, status="failed", error=message[:4000])
        return MonthResult(period, "failed", error=message)


def ingest_aemo_range(
    engine: Engine,
    cfg: AppConfig,
    start: dt.date,
    end: dt.date,
    regions: Sequence[str],
    tables: Sequence[ForecastTable] = FORECAST_TABLES,
    client: NemwebClient | None = None,
) -> dict[str, list[MonthResult]]:
    """Ingest every month in ``[start, end]`` for each table; continues past failed months."""
    for region in regions:
        cfg.region(region)
    own_client = client is None
    nemweb = client or NemwebClient(cfg.sources.mmsdm, cfg.data_path(cfg.paths.raw_dir))
    try:
        return {
            t.source: [
                ingest_forecast_month(engine, nemweb, cfg, t, m, regions)
                for m in iter_months(start, end)
            ]
            for t in tables
        }
    finally:
        if own_client:
            nemweb.close()
