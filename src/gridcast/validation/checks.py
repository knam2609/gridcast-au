"""Data-quality checks: SQL queries in ``sql/checks`` that return violating rows.

Each check's status is ``pass`` when it returns no rows, otherwise its severity
(``fail`` for data that must be fixed/investigated, ``warn`` for informational findings
such as real market events). Results are persisted to ``dq_results``.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from sqlalchemy import Engine, text

from gridcast.config import PROJECT_ROOT, AppConfig, MarketLimits
from gridcast.timeutils import NEM_TZ, month_bounds_utc, month_hours_utc

# Stand-in for "from the start of the data" in expected-missing ranges.
_BEGINNING = dt.datetime(1900, 1, 1, tzinfo=dt.UTC)

CHECKS_DIR = PROJECT_ROOT / "sql" / "checks"
MAX_EXAMPLES = 20

Severity = Literal["fail", "warn"]


@dataclass(frozen=True, slots=True)
class DQCheck:
    name: str
    severity: Severity
    description: str
    count_column: str | None = None  # sum this column for n_violations (default: row count)
    region_scoped: bool = True
    sql_name: str | None = None  # SQL file stem when several checks share one query
    params: Mapping[str, Any] = field(default_factory=dict)  # extra bind parameters

    @property
    def sql(self) -> str:
        return (CHECKS_DIR / f"{self.sql_name or self.name}.sql").read_text()


MARKET_CHECKS: tuple[DQCheck, ...] = (
    DQCheck("ingestion_failures", "fail", "Months whose latest ingestion failed", None, False),
    DQCheck("missing_intervals", "fail", "Missing 5-min intervals (gaps)", "n_missing"),
    DQCheck("rows_per_month", "fail", "Months with row count != 288 x days"),
    DQCheck("duplicate_intervals", "fail", "Duplicate intervals in the database", "n"),
    DQCheck("implausible_timestamps", "fail", "Off-grid / future / out-of-window timestamps"),
    DQCheck("null_values", "fail", "Intervals missing demand or price"),
    DQCheck("demand_out_of_range", "fail", "Demand outside plausible bounds"),
    DQCheck("price_out_of_bounds", "fail", "Price below floor / above FY market price cap"),
    DQCheck("stale_feed", "fail", "Latest interval older than freshness threshold"),
    DQCheck(
        "source_duplicates", "warn", "Duplicates removed from source", "duplicates_dropped", False
    ),
    DQCheck("market_flags", "warn", "Administered pricing / suspension episodes", "n_intervals"),
    DQCheck("price_adjustments", "warn", "Intervals with APCFLAG price post-processing bits"),
    DQCheck("price_not_firm", "warn", "Prices not marked FIRM", "n"),
)

WEATHER_CHECKS: tuple[DQCheck, ...] = (
    DQCheck("weather_site_weights", "fail", "Region has no weather sites or weights != 1"),
    DQCheck(
        "weather_gaps",
        "fail",
        "Unexpected missing hourly weather values (gaps)",
        "n_missing",
        sql_name="weather_missing",
        params={"want_known": False},
    ),
    DQCheck(
        "weather_known_gaps",
        "warn",
        "Known archive gaps (config weather.forecasts.expected_missing)",
        "n_missing",
        sql_name="weather_missing",
        params={"want_known": True},
    ),
    DQCheck("weather_out_of_range", "fail", "Weather values outside plausible bounds"),
    DQCheck("weather_run_timing", "fail", "Forecast run_init/available_at break the rule"),
    DQCheck("weather_stale", "fail", "Weather ends before the latest market interval"),
    DQCheck(
        "weather_clipped_values",
        "warn",
        "Rounding artefacts clipped onto the physical range on load",
        "n_clipped",
        region_scoped=False,
    ),
)

AEMO_CHECKS: tuple[DQCheck, ...] = (
    DQCheck("aemo_missing_runs", "warn", "AEMO runs missing from the cadence", "n_missing"),
    DQCheck("aemo_run_timing", "warn", "AEMO runs published outside the expected window"),
    DQCheck(
        "aemo_p5min_alignment",
        "fail",
        "Months where most 5-min pre-dispatch interval-0 values differ from dispatch demand",
        "n_differ",
    ),
    DQCheck("aemo_out_of_range", "fail", "AEMO forecast demand outside plausible bounds"),
)

ALL_CHECKS: tuple[DQCheck, ...] = MARKET_CHECKS + WEATHER_CHECKS + AEMO_CHECKS


@dataclass(slots=True)
class DQResult:
    check_name: str
    region_id: str | None
    status: Literal["pass", "warn", "fail"]
    n_violations: int
    details: dict[str, Any] = field(default_factory=dict)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    return value


def check_params(
    cfg: AppConfig,
    limits: MarketLimits,
    region_id: str | None,
    start_month: dt.date,
    end_month: dt.date,
) -> dict[str, Any]:
    window_start, _ = month_bounds_utc(start_month.year, start_month.month)
    _, window_end = month_bounds_utc(end_month.year, end_month.month)
    fy_years = sorted(int(k[:4]) for k in limits.market_price_cap)
    params: dict[str, Any] = {
        "region_id": region_id,
        "start_month": start_month,
        "end_month": end_month,
        "window_start": window_start.to_pydatetime(),
        "window_end": window_end.to_pydatetime(),
        "price_floor": limits.market_floor,
        "fy_start_years": fy_years,
        "fy_caps": [limits.market_price_cap[f"{y}-{(y + 1) % 100:02d}"].value for y in fy_years],
        "stale_after_days": cfg.sources.mmsdm.stale_after_days,
        **weather_params(cfg, start_month, end_month),
        **aemo_params(cfg, start_month, end_month),
    }
    if region_id is not None:
        lo, hi = cfg.region(region_id).demand_plausible_mw
        params |= {"demand_min": lo, "demand_max": hi}
    return params


def weather_params(cfg: AppConfig, start_month: dt.date, end_month: dt.date) -> dict[str, Any]:
    """Bind parameters for the weather checks (config values flattened into arrays)."""
    w = cfg.weather
    first_hour, _ = month_hours_utc(start_month.year, start_month.month)
    _, last_hour = month_hours_utc(end_month.year, end_month.month)
    known = [
        (gap.reason, var, gap.offset_days, gap.first or _BEGINNING, gap.last)
        for gap in w.forecasts.expected_missing
        for var in gap.variables
    ]
    return {
        "weather_first_hour": first_hour.to_pydatetime(),
        "weather_last_hour": last_hour.to_pydatetime(),
        "fc_model": w.forecasts.model,
        "obs_model": w.observations.model,
        "fc_offsets": list(w.forecasts.offsets_days),
        "known_reason": [k[0] for k in known],
        "known_variable": [k[1] for k in known],
        "known_offset": [k[2] for k in known],
        "known_first": [k[3] for k in known],
        "known_last": [k[4] for k in known],
        "plausible_variable": list(w.variables),
        "plausible_lo": [v.plausible[0] for v in w.variables.values()],
        "plausible_hi": [v.plausible[1] for v in w.variables.values()],
        "run_every_hours": w.forecasts.run_every_hours,
        "publication_delay_seconds": w.forecasts.publication_delay.total_seconds(),
    }


def aemo_params(cfg: AppConfig, start_month: dt.date, end_month: dt.date) -> dict[str, Any]:
    """Expected AEMO run grids (UTC) over the AEMO archive months inside the check window.

    Archive month M holds pre-dispatch runs for the trading days of M (04:00 NEM time on the
    1st to 03:30 on the 1st of M+1) and 5-minute runs from 00:05 on the 1st to 00:00 on the
    1st of M+1. Outside the AEMO window the grids are empty (first > last).
    """
    first = max(start_month, _month(cfg.aemo.ingest_start_month))
    last = min(end_month, _month(cfg.aemo.ingest_end_month))
    begin = dt.datetime(first.year, first.month, 1, tzinfo=NEM_TZ).astimezone(dt.UTC)
    after = dt.datetime(last.year, last.month, 28, tzinfo=NEM_TZ) + dt.timedelta(days=4)
    end = after.replace(day=1).astimezone(dt.UTC)
    if first > last:
        end = begin - dt.timedelta(days=1)
    return {
        "aemo_pd_first": begin + dt.timedelta(hours=4),
        "aemo_pd_last": end + dt.timedelta(hours=4) - dt.timedelta(minutes=30),
        "aemo_p5_first": begin + dt.timedelta(minutes=5),
        "aemo_p5_last": end,
        "aemo_pd_within": cfg.aemo.predispatch_published_within_minutes,
        "aemo_p5_within": cfg.aemo.p5min_published_within_minutes,
    }


def _month(value: str) -> dt.date:
    return dt.datetime.strptime(value, "%Y-%m").date()  # noqa: DTZ007 - date only


def run_check(engine: Engine, check: DQCheck, params: dict[str, Any]) -> DQResult:
    with engine.connect() as conn:
        result = conn.execute(text(check.sql), {**params, **check.params})
        columns = list(result.keys())
        rows = [dict(zip(columns, r, strict=True)) for r in result]
    col = check.count_column
    n = sum(int(r[col] or 0) for r in rows) if col else len(rows)
    status: Literal["pass", "warn", "fail"] = "pass" if n == 0 else check.severity
    details = {
        "description": check.description,
        "n_result_rows": len(rows),
        "examples": [{k: _jsonable(v) for k, v in r.items()} for r in rows[:MAX_EXAMPLES]],
    }
    return DQResult(check.name, params.get("region_id"), status, n, details)


def persist_results(engine: Engine, results: Sequence[DQResult]) -> uuid.UUID:
    check_run_id = uuid.uuid4()
    with engine.begin() as conn:
        for r in results:
            conn.execute(
                text(
                    """
                    INSERT INTO dq_results
                        (check_run_id, check_name, region_id, status, n_violations, details)
                    VALUES (:run, :name, :region, :status, :n, CAST(:details AS jsonb))
                    """
                ),
                {
                    "run": check_run_id,
                    "name": r.check_name,
                    "region": r.region_id,
                    "status": r.status,
                    "n": r.n_violations,
                    "details": json.dumps(r.details),
                },
            )
    return check_run_id


def run_checks(
    engine: Engine,
    cfg: AppConfig,
    limits: MarketLimits,
    regions: Sequence[str],
    start_month: dt.date,
    end_month: dt.date,
    checks: Sequence[DQCheck] = ALL_CHECKS,
    persist: bool = True,
) -> list[DQResult]:
    """Run every check (region-scoped ones once per region) and optionally persist results."""
    results: list[DQResult] = []
    for check in checks:
        scopes: Sequence[str | None] = regions if check.region_scoped else [None]
        for region_id in scopes:
            params = check_params(cfg, limits, region_id, start_month, end_month)
            results.append(run_check(engine, check, params))
    if persist:
        persist_results(engine, results)
    return results
