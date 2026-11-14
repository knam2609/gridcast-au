"""Regional supply totals from ``DISPATCHREGIONSUM`` → ``generation_intervals`` (M4 step 1b).

Evidence for anomaly events and inputs of the Isolation Forest (``docs/plans/m4_plan.md``
§2.3, §2.5), never a demand-model input. The monthly files are the ones ``gridcast ingest``
already cached, re-parsed for other columns (no new downloads). Days the monthly archive lacks
(1 Oct 2026) come from NEMWeb's daily dispatch reports, after an exact match with MMSDM over a
month present in both, as for market intervals (``ingestion/daily.py``).

Loader rules:

* columns are mapped by name, and each accepted table version must carry its own set:
  version 5 (Oct 2021 → Apr 2024) has semi-scheduled wind and solar cleared MW and UIGF, and
  dispatchable generation; versions 8 and 9 (May 2024 on) add ``SS_*_AVAILABILITY``, which
  must be non-blank on every row, and the bidirectional-unit (battery) totals, which may be
  blank (they are before 22 May 2024). A file with another or several versions fails the month;
* rows are the pricing run (``INTERVENTION = 0``), deduplicated as market intervals are (the
  latest ``LASTCHANGED``, then ``RUNNO``);
* a blank wind, solar or dispatchable value fails the month. A battery row is kept only where
  generation and load are both present; one without the other fails the month;
* ``TOTALINTERMITTENTGENERATION`` is an allowance in AEMO's demand forecast, not an output, and
  is not read.

Negative values and cleared output above UIGF are data-quality findings (``gridcast dq``), not
load errors.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from gridcast.config import AppConfig
from gridcast.database.load import (
    GENERATION_COLUMNS,
    ensure_regions,
    finish_ingestion_run,
    start_ingestion_run,
    upsert_generation,
)
from gridcast.ingestion.daily import SOURCE as DAILY_SOURCE
from gridcast.ingestion.daily import _same, day_interval_ends, month_days
from gridcast.ingestion.dispatch import KEY_COLUMNS, TableSpec, standardise
from gridcast.ingestion.mms_csv import MmsFormatError, ReportKey, read_mms_report
from gridcast.ingestion.nemweb import DownloadedFile, NemwebClient
from gridcast.ingestion.pipeline import SOURCE, MonthResult, _file_records, iter_months
from gridcast.logs import get_logger

log = get_logger(__name__)

TABLE = "DISPATCHREGIONSUM"
DATASET = "generation_intervals"
KEY_SECTION = ReportKey("DISPATCH", "REGIONSUM")

# Columns of every accepted version -> wide name.
BASE: dict[str, str] = {
    "SS_WIND_CLEAREDMW": "wind_cleared_mw",
    "SS_WIND_UIGF": "wind_uigf_mw",
    "SS_SOLAR_CLEAREDMW": "solar_cleared_mw",
    "SS_SOLAR_UIGF": "solar_uigf_mw",
    "DISPATCHABLEGENERATION": "dispatchable_mw",
}
# Columns added in table version 8 (May 2024) -> wide name.
LATER: dict[str, str] = {
    "SS_WIND_AVAILABILITY": "wind_available_mw",
    "SS_SOLAR_AVAILABILITY": "solar_available_mw",
    "BDU_CLEAREDMW_GEN": "battery_gen_mw",
    "BDU_CLEAREDMW_LOAD": "battery_load_mw",
}
# The LATER columns each accepted version must carry, and those that must also be non-blank.
VERSIONS: dict[str, frozenset[str]] = {
    "5": frozenset(),
    "8": frozenset(LATER),
    "9": frozenset(LATER),
}
NON_BLANK_LATER = frozenset({"SS_WIND_AVAILABILITY", "SS_SOLAR_AVAILABILITY"})
SPEC = TableSpec(name=TABLE, key=KEY_SECTION, numeric={**BASE, **LATER})

KEY = ["region_id", "interval_end", "kind"]
VALUE_COLUMNS = ("cleared_mw", "uigf_mw", "available_mw", "load_mw")
REQUIRED_KINDS = ("wind", "solar", "dispatchable")  # every interval; battery where present
DAY_INTERVALS = 288


@dataclass(slots=True)
class GenerationExtract:
    frame: pd.DataFrame  # long rows, GENERATION_COLUMNS
    rows_parsed: int
    intervention_rows_dropped: int
    duplicates_dropped: int
    versions: set[str]


def read_file(path: Path, regions: Collection[str]) -> tuple[pd.DataFrame, str]:
    """Raw string rows of one file, after checking its table version carries what it must."""
    frame, stats = read_mms_report(
        path, KEY_SECTION, [*KEY_COLUMNS, *BASE], region_filter=regions, optional=list(LATER)
    )
    if len(stats.versions) != 1:
        raise MmsFormatError(f"{path.name}: {TABLE} versions {sorted(stats.versions)} in one file")
    (version,) = stats.versions
    if version not in VERSIONS:
        raise MmsFormatError(
            f"{path.name}: unsupported {TABLE} version {version} (accepted {sorted(VERSIONS)}); "
            "check the column mapping first"
        )
    lacking = VERSIONS[version] & stats.absent_optional
    if lacking:
        raise MmsFormatError(f"{path.name}: {TABLE} v{version} lacks {sorted(lacking)}")
    pricing = frame["INTERVENTION"].str.strip().eq("0")
    for col in sorted(NON_BLANK_LATER & VERSIONS[version]):
        blank = pricing & frame[col].str.strip().fillna("").eq("")
        if blank.any():
            raise MmsFormatError(f"{path.name}: {TABLE} v{version}: {int(blank.sum())} blank {col}")
    return frame, version


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """One row per (region, interval, kind) from the standardised wide frame."""
    required = wide[list(BASE.values())]
    if required.isna().any().any():
        counts = required.isna().sum()
        raise MmsFormatError(f"{TABLE}: blank values {counts[counts > 0].to_dict()}")
    gen, load = wide["battery_gen_mw"], wide["battery_load_mw"]
    lopsided = gen.isna() != load.isna()
    if lopsided.any():
        sample = wide.loc[lopsided, "interval_end"].head(3).tolist()
        raise MmsFormatError(
            f"{TABLE}: {int(lopsided.sum())} rows with only one of BDU generation and load, "
            f"e.g. {sample}"
        )
    key = wide[["region_id", "interval_end"]]
    nan = np.nan
    parts = [
        key.assign(
            kind=kind,
            cleared_mw=wide[f"{kind}_cleared_mw"],
            uigf_mw=wide[f"{kind}_uigf_mw"],
            available_mw=wide[f"{kind}_available_mw"],
            load_mw=nan,
        )
        for kind in ("wind", "solar")
    ]
    parts.append(
        key.assign(
            kind="dispatchable",
            cleared_mw=wide["dispatchable_mw"],
            uigf_mw=nan,
            available_mw=nan,
            load_mw=nan,
        )
    )
    has = gen.notna()
    parts.append(
        key[has].assign(
            kind="battery", cleared_mw=gen[has], uigf_mw=nan, available_mw=nan, load_mw=load[has]
        )
    )
    out = pd.concat(parts, ignore_index=True)
    out["kind"] = out["kind"].astype("string")
    for col in VALUE_COLUMNS:
        out[col] = out[col].astype("float64")
    return out.sort_values(KEY).reset_index(drop=True)[list(GENERATION_COLUMNS)]


def extract_generation(paths: Sequence[Path], regions: Collection[str]) -> GenerationExtract:
    """Parse, version-check and standardise all parts of one month (or one daily archive)."""
    frames: list[pd.DataFrame] = []
    versions: set[str] = set()
    for path in paths:
        frame, version = read_file(path, regions)
        frames.append(frame)
        versions.add(version)
    columns = [*KEY_COLUMNS, *BASE, *LATER]
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    wide, n_intervention, n_dupes = standardise(raw, SPEC)
    return GenerationExtract(to_long(wide), len(raw), n_intervention, n_dupes, versions)


def ingest_generation_month(
    engine: Engine,
    client: NemwebClient,
    cfg: AppConfig,
    period: dt.date,
    regions: Sequence[str],
) -> MonthResult:
    """Load one MMSDM month of supply totals; failures are recorded, not raised."""
    with engine.begin() as conn:
        ensure_regions(conn, cfg)
        run_id = start_ingestion_run(conn, SOURCE, DATASET, period)
    bound = log.bind(table=TABLE, period=f"{period:%Y-%m}", run_id=run_id)
    downloads: list[DownloadedFile] = []
    try:
        downloads = client.fetch_month(TABLE, period.year, period.month)
        extract = extract_generation([d.path for d in downloads], regions)
        interim = cfg.data_path(cfg.paths.interim_dir) / DATASET / f"{period:%Y%m}.parquet"
        interim.parent.mkdir(parents=True, exist_ok=True)
        extract.frame.to_parquet(interim, index=False)
        with engine.begin() as conn:
            loaded = upsert_generation(conn, extract.frame, run_id)
            finish_ingestion_run(
                conn,
                run_id,
                status="success",
                files=_file_records(downloads),
                rows_parsed=extract.rows_parsed,
                duplicates_dropped=extract.duplicates_dropped,
                rows_loaded=loaded,
            )
        bound.info(
            "ingest_generation.month_ok",
            rows=len(extract.frame),
            rows_loaded=loaded,
            kinds=extract.frame["kind"].value_counts().to_dict(),
            intervention_rows_dropped=extract.intervention_rows_dropped,
            versions=sorted(extract.versions),
        )
        return MonthResult(
            period, "success", extract.rows_parsed, extract.duplicates_dropped, loaded
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        bound.error("ingest_generation.month_failed", error=message)
        with engine.begin() as conn:
            finish_ingestion_run(
                conn, run_id, status="failed", files=_file_records(downloads), error=message[:4000]
            )
        return MonthResult(period, "failed", error=message)


def ingest_generation_range(
    engine: Engine,
    cfg: AppConfig,
    start: dt.date,
    end: dt.date,
    regions: Sequence[str],
    client: NemwebClient | None = None,
) -> list[MonthResult]:
    """Load every month in ``[start, end]``; continues past failed months."""
    for region in regions:
        cfg.region(region)
    own_client = client is None
    nemweb = client or NemwebClient(cfg.sources.mmsdm, cfg.data_path(cfg.paths.raw_dir))
    try:
        return [
            ingest_generation_month(engine, nemweb, cfg, m, regions)
            for m in iter_months(start, end)
        ]
    finally:
        if own_client:
            nemweb.close()


# --- Days MMSDM lacks, from the daily dispatch reports -------------------------------------


def transform_day(path: Path, day: dt.date, regions: Sequence[str]) -> tuple[pd.DataFrame, int]:
    """Long rows of one daily archive; every required kind must have the day's 288 intervals."""
    extract = extract_generation([path], regions)
    frame = extract.frame
    expected = day_interval_ends(day)
    for region in regions:
        for kind in REQUIRED_KINDS:
            got = frame.loc[(frame["region_id"] == region) & (frame["kind"] == kind)]
            if not pd.DatetimeIndex(got["interval_end"]).equals(expected):
                raise ValueError(
                    f"{path.name}: {region} {kind} has {len(got)} intervals, expected the "
                    f"{DAY_INTERVALS} of {day} (NEM time)"
                )
    return frame, extract.rows_parsed


def load_existing(
    engine: Engine, regions: Sequence[str], first_end: pd.Timestamp, last_end: pd.Timestamp
) -> pd.DataFrame:
    """Loaded ``generation_intervals`` rows in ``[first_end, last_end]`` with their source."""
    cols = ", ".join(f"g.{c}" for c in GENERATION_COLUMNS)
    with engine.connect() as conn:
        frame = pd.read_sql(
            text(
                f"""
                SELECT {cols}, r.source
                FROM generation_intervals g
                LEFT JOIN ingestion_runs r ON r.run_id = g.ingestion_run_id
                WHERE g.region_id = ANY(:regions)
                  AND g.interval_end BETWEEN :first AND :last
                ORDER BY g.region_id, g.interval_end, g.kind
                """
            ),
            conn,
            params={
                "regions": list(regions),
                "first": first_end.to_pydatetime(),
                "last": last_end.to_pydatetime(),
            },
        )
    frame["interval_end"] = pd.to_datetime(frame["interval_end"], utc=True)
    return frame


@dataclass(slots=True)
class GenerationMatchCheck:
    """The daily reports against MMSDM over one month: every column must match exactly."""

    month: str
    days: int
    regions: list[str]
    loaded_sources: list[str]
    compared: int
    only_daily: int
    only_loaded: int
    differing: dict[str, int]

    @property
    def passed(self) -> bool:
        return (
            self.compared > 0
            and self.only_daily == 0
            and self.only_loaded == 0
            and self.loaded_sources == [SOURCE]
            and not any(self.differing.values())
        )

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def compare_generation(
    daily: pd.DataFrame, loaded: pd.DataFrame
) -> tuple[int, int, int, dict[str, int]]:
    """``(shared rows, only daily, only loaded, differing values per column)``."""
    merged = daily[list(GENERATION_COLUMNS)].merge(
        loaded[list(GENERATION_COLUMNS)],
        on=KEY,
        how="outer",
        suffixes=("_daily", "_loaded"),
        indicator=True,
        validate="one_to_one",
    )
    both = merged[merged["_merge"] == "both"]
    differing = {
        col: int((~_same(both[f"{col}_daily"], both[f"{col}_loaded"])).sum())
        for col in VALUE_COLUMNS
    }
    only_daily = int((merged["_merge"] == "left_only").sum())
    only_loaded = int((merged["_merge"] == "right_only").sum())
    return len(both), only_daily, only_loaded, differing


@dataclass(slots=True)
class GenerationDailyResult:
    check: GenerationMatchCheck
    loaded: dict[str, int] = field(default_factory=dict)  # NEM day -> rows loaded
    run_ids: list[int] = field(default_factory=list)
    report_path: Path | None = None


class GenerationCheckFailed(RuntimeError):
    def __init__(self, message: str, check: GenerationMatchCheck) -> None:
        super().__init__(message)
        self.check = check


def ingest_generation_daily(
    engine: Engine,
    cfg: AppConfig,
    days: Sequence[dt.date],
    check_month: dt.date,
    regions: Sequence[str],
    client: NemwebClient | None = None,
) -> GenerationDailyResult:
    """Check the daily reports against MMSDM over ``check_month``, then load ``days``.

    The same rules as ``ingest_daily`` for market intervals, except that every generation
    column must match exactly (it did in every September 2026 interval when this was written).
    """
    if cfg.sources.nemweb_daily is None:
        raise ValueError("sources.nemweb_daily is not configured")
    if not days:
        raise ValueError("no days to load")
    check_days = month_days(check_month)
    if set(days) & set(check_days):
        raise ValueError("the days to load must lie outside the check month")
    for region in regions:
        cfg.region(region)
    own_client = client is None
    nemweb = client or NemwebClient(cfg.sources.mmsdm, cfg.data_path(cfg.paths.raw_dir))
    try:
        files = nemweb.fetch_daily_reports(
            cfg.sources.nemweb_daily.listing_url, [*check_days, *sorted(days)]
        )
    finally:
        if own_client:
            nemweb.close()

    daily = pd.concat(
        [transform_day(files[d].path, d, regions)[0] for d in check_days], ignore_index=True
    )
    loaded = load_existing(
        engine,
        regions,
        day_interval_ends(check_days[0])[0],
        day_interval_ends(check_days[-1])[-1],
    )
    compared, only_daily, only_loaded, differing = compare_generation(daily, loaded)
    sources = sorted(str(s) for s in loaded["source"].dropna().unique())
    if loaded["source"].isna().any():
        sources.append("<no ingestion run>")
    check = GenerationMatchCheck(
        month=f"{check_month:%Y-%m}",
        days=len(check_days),
        regions=list(regions),
        loaded_sources=sources,
        compared=compared,
        only_daily=only_daily,
        only_loaded=only_loaded,
        differing=differing,
    )
    result = GenerationDailyResult(check)
    report = (
        cfg.data_path(cfg.paths.interim_dir)
        / DAILY_SOURCE
        / f"generation_check_{check_month:%Y-%m}.json"
    )
    result.report_path = report

    def write_report() -> None:
        report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checked_at": dt.datetime.now(dt.UTC).isoformat(),
            "check": check.as_dict(),
            "files": _file_records([files[d] for d in check_days]),
            "loaded": result.loaded,
            "ingestion_run_ids": result.run_ids,
        }
        report.write_text(json.dumps(payload, indent=2))

    log.info("generation_daily.check", month=check.month, passed=check.passed)
    if not check.passed:
        write_report()
        raise GenerationCheckFailed(
            f"daily reports disagree with MMSDM over {check.month}; nothing loaded (see {report})",
            check,
        )

    for month in iter_months(min(days), max(days)):
        month_list = sorted(d for d in days if (d.year, d.month) == (month.year, month.month))
        if not month_list:
            continue
        parts = [transform_day(files[d].path, d, regions) for d in month_list]
        frame = pd.concat([p[0] for p in parts], ignore_index=True)
        first = day_interval_ends(month_list[0])[0]
        last = day_interval_ends(month_list[-1])[-1]
        if (load_existing(engine, regions, first, last)["source"] == SOURCE).any():
            raise ValueError(
                f"{month:%Y-%m}: some intervals are already loaded from MMSDM, which stays "
                "authoritative; nothing loaded"
            )
        with engine.begin() as conn:
            ensure_regions(conn, cfg)
            run_id = start_ingestion_run(conn, DAILY_SOURCE, DATASET, month)
        downloads = [files[d] for d in month_list]
        try:
            with engine.begin() as conn:
                n = upsert_generation(conn, frame, run_id)
                finish_ingestion_run(
                    conn,
                    run_id,
                    status="success",
                    files=_file_records(downloads),
                    rows_parsed=sum(p[1] for p in parts),
                    duplicates_dropped=None,
                    rows_loaded=n,
                )
        except Exception as exc:
            with engine.begin() as conn:
                finish_ingestion_run(
                    conn,
                    run_id,
                    status="failed",
                    files=_file_records(downloads),
                    error=f"{type(exc).__name__}: {exc}"[:4000],
                )
            raise
        result.run_ids.append(run_id)
        for d, (part, _) in zip(month_list, parts, strict=True):
            result.loaded[d.isoformat()] = len(part)
        log.info("generation_daily.loaded", month=f"{month:%Y-%m}", days=len(month_list), rows=n)
    write_report()
    return result
