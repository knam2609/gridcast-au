"""Market intervals from NEMWeb's daily dispatch reports, checked against MMSDM (M2 step 6).

The monthly MMSDM archive appears two to six weeks after month end. The M2 holdout's last
issue times (30 Sep 2026) have targets on 1 Oct 2026, so that day is read from NEMWeb's daily
archive of the five-minute dispatch reports instead. They hold the same ``DISPATCH.REGIONSUM``
and ``DISPATCH.PRICE`` records, parsed and validated exactly as the monthly tables are
(m2_plan.md step 6, decision 1).

* **Match check, first:** every day of ``check_month``, already loaded from MMSDM, is read from
  the daily archive and every loaded column is compared with the database at every interval.
  Demand (``total_demand_mw``) must be identical everywhere, and no interval may be missing
  on either side; otherwise nothing is loaded. The other columns' differences are reported.
* **Load:** each requested day must be complete (288 intervals per region) and not already
  loaded from MMSDM, which stays authoritative: re-ingesting the month from MMSDM later
  rewrites any changed value. Loads are recorded in ``ingestion_runs`` (source
  ``nemweb_daily``).
* The comparison is written to ``<data root>/interim/nemweb_daily/check_<YYYY-MM>.json`` so
  that reports quote it instead of retyping it.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from gridcast.config import AppConfig
from gridcast.database.load import (
    MARKET_COLUMNS,
    ensure_regions,
    finish_ingestion_run,
    start_ingestion_run,
    upsert_market_intervals,
)
from gridcast.ingestion.nemweb import DownloadedFile, NemwebClient
from gridcast.ingestion.pipeline import DATASET, _file_records, iter_months, transform_month
from gridcast.logs import get_logger
from gridcast.timeutils import INTERVAL, NEM_TZ

log = get_logger(__name__)

SOURCE = "nemweb_daily"
KEY = ["region_id", "interval_end"]
VALUE_COLUMNS: tuple[str, ...] = MARKET_COLUMNS[3:]
MUST_MATCH = ("total_demand_mw",)
DAY_INTERVALS = 288


class DailyCheckFailed(RuntimeError):
    """The daily reports disagree with MMSDM on a column that must match."""

    def __init__(self, message: str, check: MatchCheck) -> None:
        super().__init__(message)
        self.check = check


def day_interval_ends(day: dt.date) -> pd.DatetimeIndex:
    """UTC ends of the 288 intervals of NEM-time ``day`` (00:05 → 24:00)."""
    start = pd.Timestamp(dt.datetime.combine(day, dt.time(), tzinfo=NEM_TZ)).tz_convert("UTC")
    return pd.date_range(start + INTERVAL, periods=DAY_INTERVALS, freq=INTERVAL)


def month_days(month: dt.date) -> list[dt.date]:
    first = month.replace(day=1)
    nxt = (first.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return [first + dt.timedelta(days=i) for i in range((nxt - first).days)]


def transform_day(path: Path, day: dt.date, regions: Sequence[str]) -> tuple[pd.DataFrame, int]:
    """Validated ``market_intervals`` rows of one daily archive; returns (rows, rows parsed)."""
    frame, demand, price = transform_month([path], [path], regions)
    expected = day_interval_ends(day)
    for region in regions:
        got = pd.DatetimeIndex(frame.loc[frame["region_id"] == region, "interval_end"])
        if not got.equals(expected):
            raise ValueError(
                f"{path.name}: {region} has {len(got)} intervals, expected the "
                f"{DAY_INTERVALS} of {day} (NEM time)"
            )
    return frame, demand.rows_parsed + price.rows_parsed


def load_existing(
    engine: Engine, regions: Sequence[str], first_end: pd.Timestamp, last_end: pd.Timestamp
) -> pd.DataFrame:
    """Loaded ``market_intervals`` rows in ``[first_end, last_end]`` with their source."""
    cols = ", ".join(f"m.{c}" for c in ("region_id", "interval_end", *VALUE_COLUMNS))
    with engine.connect() as conn:
        frame = pd.read_sql(
            text(
                f"""
                SELECT {cols}, r.source
                FROM market_intervals m
                LEFT JOIN ingestion_runs r ON r.run_id = m.ingestion_run_id
                WHERE m.region_id = ANY(:regions)
                  AND m.interval_end BETWEEN :first AND :last
                ORDER BY m.region_id, m.interval_end
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


@dataclass(frozen=True, slots=True)
class ColumnComparison:
    column: str
    compared: int
    differing: int
    max_abs_difference: float | None  # numeric columns only

    @property
    def must_match(self) -> bool:
        return self.column in MUST_MATCH


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Element-wise exact equality, with missing equal to missing."""
    left, right = a.astype(object), b.astype(object)
    a_na, b_na = left.isna().to_numpy(), right.isna().to_numpy()
    equal = np.array(
        [
            bool(x == y) if not (xn or yn) else False
            for x, y, xn, yn in zip(left, right, a_na, b_na, strict=True)
        ],
        dtype=bool,
    )
    return np.where(a_na | b_na, a_na & b_na, equal)


def compare_frames(
    daily: pd.DataFrame, loaded: pd.DataFrame
) -> tuple[list[ColumnComparison], int, int]:
    """Per-column comparison on shared intervals, plus intervals only in daily / only loaded."""
    merged = daily[[*KEY, *VALUE_COLUMNS]].merge(
        loaded[[*KEY, *VALUE_COLUMNS]],
        on=KEY,
        how="outer",
        suffixes=("_daily", "_loaded"),
        indicator=True,
        validate="one_to_one",
    )
    both = merged[merged["_merge"] == "both"]
    out = []
    for col in VALUE_COLUMNS:
        a, b = both[f"{col}_daily"], both[f"{col}_loaded"]
        same = _same(a, b)
        diff: float | None = None
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            gaps = (a.astype("float64") - b.astype("float64")).abs()
            diff = float(gaps.max()) if gaps.notna().any() else 0.0
        out.append(ColumnComparison(col, len(both), int((~same).sum()), diff))
    only_daily = int((merged["_merge"] == "left_only").sum())
    only_loaded = int((merged["_merge"] == "right_only").sum())
    return out, only_daily, only_loaded


@dataclass(slots=True)
class MatchCheck:
    """The daily archive against MMSDM over one month (decision 1 of m2_plan.md step 6)."""

    month: str
    days: int
    regions: list[str]
    loaded_sources: list[str]
    only_daily: int
    only_loaded: int
    columns: list[ColumnComparison]

    @property
    def passed(self) -> bool:
        return (
            self.only_daily == 0
            and self.only_loaded == 0
            and self.loaded_sources == ["mmsdm"]
            and all(c.differing == 0 for c in self.columns if c.must_match)
        )

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed, "must_match": list(MUST_MATCH)}


@dataclass(slots=True)
class DailyResult:
    check: MatchCheck
    loaded: dict[str, int] = field(default_factory=dict)  # NEM day -> rows loaded
    run_ids: list[int] = field(default_factory=list)
    report_path: Path | None = None


def run_match_check(
    engine: Engine,
    files: dict[dt.date, DownloadedFile],
    month: dt.date,
    regions: Sequence[str],
) -> MatchCheck:
    days = month_days(month)
    frames = [transform_day(files[d].path, d, regions)[0] for d in days]
    daily = pd.concat(frames, ignore_index=True)
    loaded = load_existing(
        engine, regions, day_interval_ends(days[0])[0], day_interval_ends(days[-1])[-1]
    )
    columns, only_daily, only_loaded = compare_frames(daily, loaded)
    sources = sorted(str(s) for s in loaded["source"].dropna().unique())
    if loaded["source"].isna().any():
        sources.append("<no ingestion run>")
    return MatchCheck(
        month=f"{month:%Y-%m}",
        days=len(days),
        regions=list(regions),
        loaded_sources=sources,
        only_daily=only_daily,
        only_loaded=only_loaded,
        columns=columns,
    )


def ingest_daily(
    engine: Engine,
    cfg: AppConfig,
    days: Sequence[dt.date],
    check_month: dt.date,
    regions: Sequence[str],
    client: NemwebClient | None = None,
) -> DailyResult:
    """Check the daily archive against MMSDM over ``check_month``, then load ``days``."""
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
    url = cfg.sources.nemweb_daily.listing_url
    try:
        files = nemweb.fetch_daily_reports(url, [*check_days, *sorted(days)])
    finally:
        if own_client:
            nemweb.close()

    check = run_match_check(engine, files, check_month, regions)
    result = DailyResult(check)
    report = cfg.data_path(cfg.paths.interim_dir) / SOURCE / f"check_{check_month:%Y-%m}.json"
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

    log.info("daily.check", month=check.month, passed=check.passed)
    if not check.passed:
        write_report()
        raise DailyCheckFailed(
            f"daily reports disagree with MMSDM over {check.month}; nothing loaded (see {report})",
            check,
        )

    for month in iter_months(min(days), max(days)):
        month_list = sorted(d for d in days if (d.year, d.month) == (month.year, month.month))
        if not month_list:
            continue
        parts = [transform_day(files[d].path, d, regions) for d in month_list]
        frame = pd.concat([p[0] for p in parts], ignore_index=True)
        first, last = day_interval_ends(month_list[0])[0], day_interval_ends(month_list[-1])[-1]
        existing = load_existing(engine, regions, first, last)
        if (existing["source"] == "mmsdm").any():
            raise ValueError(
                f"{month:%Y-%m}: some intervals are already loaded from MMSDM, which stays "
                "authoritative; nothing loaded"
            )
        with engine.begin() as conn:
            ensure_regions(conn, cfg)
            run_id = start_ingestion_run(conn, SOURCE, DATASET, month)
        downloads = [files[d] for d in month_list]
        try:
            with engine.begin() as conn:
                loaded = upsert_market_intervals(conn, frame, run_id)
                finish_ingestion_run(
                    conn,
                    run_id,
                    status="success",
                    files=_file_records(downloads),
                    rows_parsed=sum(p[1] for p in parts),
                    duplicates_dropped=None,
                    rows_loaded=loaded,
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
        log.info("daily.loaded", month=f"{month:%Y-%m}", days=len(month_list), rows=loaded)
    write_report()
    return result
