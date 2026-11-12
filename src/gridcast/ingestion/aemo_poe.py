"""AEMO's probabilistic demand forecasts (M3 benchmark only, never a model input): every
``LOR`` run of pre-dispatch PASA (``PDPASA_REGIONSOLUTION``, MMSDM monthly ``DATA/`` folder)
→ ``aemo_demand_poe``.

Loader rules (``docs/m3_data_spike.md``, ``docs/plans/m3_plan.md`` §2.6):

* columns are mapped by name, and only table version 7 (the version of the whole benchmark
  window in step 0) is accepted; anything else fails the month;
* only ``RUNTYPE = 'LOR'`` rows are kept (``OUTAGE_LRC`` was discontinued from 31 July
  2025), and a duplicate (region, run, period) fails;
* ``run_time`` is the nominal run time ``RUN_DATETIME − 30 min``: ``RUN_DATETIME`` is the
  end of the run's first period, and this nominal time matches the pre-dispatch run with
  the same information (spike §2). Every run's first period must end at ``RUN_DATETIME``,
  and its periods must lie on the 30-minute grid after it;
* ``available_at`` is ``LASTCHANGED``, which must be the same for every row of a run;
* ``DEMAND10 ≥ DEMAND50 ≥ DEMAND90`` must hold on every row (it did on every row of the
  four months sampled in step 0).
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
    upsert_aemo_poe,
)
from gridcast.ingestion.dispatch import _to_float
from gridcast.ingestion.mms_csv import MmsFormatError, ReportKey, read_mms_report
from gridcast.ingestion.nemweb import DownloadedFile, NemwebClient
from gridcast.ingestion.pipeline import SOURCE, MonthResult, iter_months
from gridcast.logs import get_logger
from gridcast.timeutils import parse_nem_timestamps

log = get_logger(__name__)

TABLE = "PDPASA_REGIONSOLUTION"
DATASET = "aemo_pdpasa"
KEY_SECTION = ReportKey("PDPASA", "REGIONSOLUTION")
VERSIONS = frozenset({"7"})
RUN_TYPE = "LOR"
PERIOD = pd.Timedelta(minutes=30)
COLUMNS = (
    "RUN_DATETIME",
    "INTERVAL_DATETIME",
    "REGIONID",
    "RUNTYPE",
    "DEMAND10",
    "DEMAND50",
    "DEMAND90",
    "LASTCHANGED",
)
KEY = ["region_id", "run_time", "target_end"]
OUTPUT_COLUMNS = [*KEY, "available_at", "demand10_mw", "demand50_mw", "demand90_mw"]


@dataclass(slots=True)
class PoeExtract:
    frame: pd.DataFrame
    rows_parsed: int
    other_run_type_rows: int  # rows of other run types (dropped)
    versions: set[str]


def standardise(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep LOR runs, derive run/period/availability times (UTC), validate structure.

    Returns ``(frame, rows_of_other_run_types)``. Raises :class:`MmsFormatError` on a
    duplicate key, a run off the 30-minute grid, a run with several publication times, a
    blank demand or an out-of-order POE band.
    """
    keep = raw["RUNTYPE"].str.strip().eq(RUN_TYPE)
    other = int((~keep).sum())
    df = raw.loc[keep]
    first_end = parse_nem_timestamps(df["RUN_DATETIME"])
    out = pd.DataFrame(
        {
            "region_id": df["REGIONID"].str.strip().astype("string"),
            "run_time": first_end - PERIOD,
            "target_end": parse_nem_timestamps(df["INTERVAL_DATETIME"]),
            "available_at": parse_nem_timestamps(df["LASTCHANGED"]),
            **{
                f"demand{p}_mw": _to_float(df[f"DEMAND{p}"], f"{TABLE}.DEMAND{p}")
                for p in (10, 50, 90)
            },
        }
    ).reset_index(drop=True)
    demand = out[["demand10_mw", "demand50_mw", "demand90_mw"]]
    if demand.isna().any().any():
        raise MmsFormatError(f"{TABLE}: {int(demand.isna().any(axis=1).sum())} rows lack a POE")

    seconds = (out["target_end"] - out["run_time"]).dt.total_seconds()
    step = PERIOD.total_seconds()
    on_grid = (seconds % step == 0) & (seconds >= step)
    on_grid &= out["run_time"].dt.floor("30min") == out["run_time"]
    first = seconds.groupby([out["region_id"], out["run_time"]]).transform("min")
    off = ~on_grid | (first != step)
    if off.any():
        sample = out.loc[off, ["run_time", "target_end"]].head(3).to_dict("records")
        raise MmsFormatError(f"{TABLE}: {int(off.sum())} periods off the run grid: {sample}")

    published = out.groupby(["region_id", "run_time"])["available_at"].nunique()
    if (published > 1).any():
        runs = published[published > 1].index[:3].tolist()
        raise MmsFormatError(f"{TABLE}: runs with several LASTCHANGED values, e.g. {runs}")

    bad = (out["demand10_mw"] < out["demand50_mw"]) | (out["demand50_mw"] < out["demand90_mw"])
    if bad.any():
        sample = out.loc[bad, KEY].head(3).to_dict("records")
        raise MmsFormatError(f"{TABLE}: {int(bad.sum())} rows with POE10 < POE50 < POE90: {sample}")

    dupes = out.duplicated(subset=KEY, keep=False)
    if dupes.any():
        sample = out.loc[dupes, KEY].head(4).to_dict("records")
        raise MmsFormatError(f"{TABLE}: {int(dupes.sum())} rows share a key, e.g. {sample}")
    return out.sort_values(KEY).reset_index(drop=True)[OUTPUT_COLUMNS], other


def extract_poe(paths: Sequence[Path], regions: Collection[str]) -> PoeExtract:
    """Parse, check the table version of, and standardise all parts of one month."""
    frames: list[pd.DataFrame] = []
    versions: set[str] = set()
    for path in paths:
        frame, stats = read_mms_report(path, KEY_SECTION, COLUMNS, region_filter=regions)
        frames.append(frame)
        versions |= stats.versions
    unknown = versions - VERSIONS
    if unknown:
        raise MmsFormatError(
            f"{TABLE}: unsupported version(s) {sorted(unknown)} (accepted {sorted(VERSIONS)}); "
            "check the column mapping first"
        )
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    clean, other = standardise(raw)
    return PoeExtract(clean, len(raw), other, versions)


def ingest_poe_month(
    engine: Engine,
    client: NemwebClient,
    cfg: AppConfig,
    period: dt.date,
    regions: Sequence[str],
) -> MonthResult:
    """Ingest one month of PDPASA POE forecasts; failures are recorded, not raised."""
    with engine.begin() as conn:
        ensure_regions(conn, cfg)
        run_id = start_ingestion_run(conn, SOURCE, DATASET, period)
    bound = log.bind(table=TABLE, period=f"{period:%Y-%m}", run_id=run_id)
    downloads: list[DownloadedFile] = []
    try:
        downloads = client.fetch_month(TABLE, period.year, period.month)
        extract = extract_poe([d.path for d in downloads], regions)
        interim = cfg.data_path(cfg.paths.interim_dir) / DATASET / f"{period:%Y%m}.parquet"
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
            loaded = upsert_aemo_poe(conn, extract.frame, run_id)
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
            "ingest_aemo_poe.month_ok",
            rows=len(extract.frame),
            rows_loaded=loaded,
            runs=int(extract.frame.groupby(["region_id", "run_time"]).ngroups),
            other_run_type_rows=extract.other_run_type_rows,
            versions=sorted(extract.versions),
        )
        return MonthResult(period, "success", extract.rows_parsed, 0, loaded)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        bound.error("ingest_aemo_poe.month_failed", error=message)
        with engine.begin() as conn:
            finish_ingestion_run(conn, run_id, status="failed", error=message[:4000])
        return MonthResult(period, "failed", error=message)


def ingest_poe_range(
    engine: Engine,
    cfg: AppConfig,
    start: dt.date,
    end: dt.date,
    regions: Sequence[str],
    client: NemwebClient | None = None,
) -> list[MonthResult]:
    """Ingest every month in ``[start, end]``; continues past failed months."""
    for region in regions:
        cfg.region(region)
    own_client = client is None
    nemweb = client or NemwebClient(cfg.sources.mmsdm, cfg.data_path(cfg.paths.raw_dir))
    try:
        return [ingest_poe_month(engine, nemweb, cfg, m, regions) for m in iter_months(start, end)]
    finally:
        if own_client:
            nemweb.close()
