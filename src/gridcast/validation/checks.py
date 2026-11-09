"""Data-quality checks: SQL queries in ``sql/checks`` that return violating rows.

Each check's status is ``pass`` when it returns no rows, otherwise its severity
(``fail`` for data that must be fixed/investigated, ``warn`` for informational findings
such as real market events). Results are persisted to ``dq_results``.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from sqlalchemy import Engine, text

from gridcast.config import PROJECT_ROOT, AppConfig, MarketLimits
from gridcast.timeutils import month_bounds_utc

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

    @property
    def sql(self) -> str:
        return (CHECKS_DIR / f"{self.name}.sql").read_text()


CHECKS: tuple[DQCheck, ...] = (
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
    }
    if region_id is not None:
        lo, hi = cfg.region(region_id).demand_plausible_mw
        params |= {"demand_min": lo, "demand_max": hi}
    return params


def run_check(engine: Engine, check: DQCheck, params: dict[str, Any]) -> DQResult:
    with engine.connect() as conn:
        result = conn.execute(text(check.sql), params)
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
    checks: Sequence[DQCheck] = CHECKS,
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
