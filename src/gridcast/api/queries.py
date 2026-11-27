"""The SQL behind each endpoint (read-only; m5_plan.md §2.2, §2.4).

A forecast is one primary-key range of ``forecasts`` / ``forecast_intervals`` (≤ 288 rows).
Evolution for a target scans the primary key over at most 48 issue times and filters on the
target. Anomalies page by keyset on (first interval end, event id).

Bounds are ``real``: psycopg returns each as the float nearest its shortest decimal, which
:func:`real` turns back into the exact float64 of the stored float32 (m5_plan.md §1).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

BOUNDS = ("lo95", "lo80", "lo50", "hi50", "hi80", "hi95")
ANOMALY_SUMMARY = """
    event_id, run_id, split, first_interval_end, last_interval_end, peak_interval_end, tier,
    detectors, categories, min_p, lines_only, summary
"""


def _dt(t: pd.Timestamp) -> Any:
    return t.to_pydatetime()


def real(value: Any) -> float:
    """The exact float64 of a ``real`` column's float32 value."""
    return float(np.float32(value))


def points(engine: Engine, run_id: int, issue_time: pd.Timestamp) -> dict[int, float]:
    """Stored point forecasts of one issue time, by lead."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT horizon, point FROM forecasts WHERE run_id = :r AND issue_time = :t"),
            {"r": run_id, "t": _dt(issue_time)},
        ).all()
    return {int(r.horizon): float(r.point) for r in rows}


def bounds(engine: Engine, run_id: int, issue_time: pd.Timestamp) -> dict[int, tuple[float, ...]]:
    """Stored interval bounds of one issue time, by lead, in :data:`BOUNDS` order."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"SELECT horizon, {', '.join(BOUNDS)} FROM forecast_intervals "
                "WHERE run_id = :r AND issue_time = :t"
            ),
            {"r": run_id, "t": _dt(issue_time)},
        ).all()
    return {int(r.horizon): tuple(real(getattr(r, b)) for b in BOUNDS) for r in rows}


def evolution(
    engine: Engine,
    point_runs: Sequence[int],
    interval_runs: Sequence[int],
    first_issue: pd.Timestamp,
    target_end: pd.Timestamp,
) -> tuple[dict[pd.Timestamp, float], dict[pd.Timestamp, tuple[float, ...]]]:
    """Every stored point and bounds for ``target_end`` issued in [first_issue, target_end)."""
    params = {"lo": _dt(first_issue), "t": _dt(target_end)}
    with engine.connect() as conn:
        p_rows = conn.execute(
            text(
                "SELECT issue_time, point FROM forecasts WHERE run_id = ANY(:runs) "
                "AND issue_time >= :lo AND issue_time < :t AND target_end = :t"
            ),
            {**params, "runs": list(point_runs)},
        ).all()
        i_rows = conn.execute(
            text(
                f"SELECT issue_time, {', '.join(BOUNDS)} FROM forecast_intervals "
                "WHERE run_id = ANY(:runs) AND issue_time >= :lo AND issue_time < :t "
                "AND target_end = :t"
            ),
            {**params, "runs": list(interval_runs)},
        ).all()
    pts = {pd.Timestamp(r.issue_time).as_unit("us"): float(r.point) for r in p_rows}
    bnd = {
        pd.Timestamp(r.issue_time).as_unit("us"): tuple(real(getattr(r, b)) for b in BOUNDS)
        for r in i_rows
    }
    return pts, bnd


def actuals(engine: Engine, region: str, start: pd.Timestamp, end: pd.Timestamp) -> list[Any]:
    """Intervals inside [start, end]: interval_start ≥ start and interval_end ≤ end."""
    with engine.connect() as conn:
        return list(
            conn.execute(
                text(
                    """
                    SELECT interval_start, interval_end, total_demand_mw, rrp_aud_mwh,
                           price_status, apc_flag_bits, administered_price_binding,
                           market_cap_floor_binding, price_manually_overwritten,
                           market_suspended
                    FROM market_intervals
                    WHERE region_id = :r AND interval_end > :lo AND interval_end <= :hi
                      AND interval_start >= :lo
                    ORDER BY interval_end
                    """
                ),
                {"r": region, "lo": _dt(start), "hi": _dt(end)},
            ).all()
        )


def anomaly_page(
    engine: Engine,
    run_ids: Sequence[int],
    start: pd.Timestamp,
    end: pd.Timestamp,
    limit: int,
    tier: str | None = None,
    category: str | None = None,
    detector: str | None = None,
    after: tuple[pd.Timestamp, int] | None = None,
) -> list[Any]:
    """Up to ``limit`` events of the served runs whose first interval end is in [start, end),
    in (first interval end, event id) order, after the cursor position if given."""
    where = [
        "run_id = ANY(:runs)",
        "first_interval_end >= :lo",
        "first_interval_end < :hi",
    ]
    params: dict[str, Any] = {
        "runs": list(run_ids),
        "lo": _dt(start),
        "hi": _dt(end),
        "limit": limit,
    }
    if tier is not None:
        where.append("tier = :tier")
        params["tier"] = tier
    if category is not None:
        where.append(":category = ANY(categories)")
        params["category"] = category
    if detector is not None:
        where.append(":detector = ANY(detectors)")
        params["detector"] = detector
    if after is not None:
        where.append("(first_interval_end, event_id) > (:after_t, :after_id)")
        params["after_t"], params["after_id"] = _dt(after[0]), after[1]
    sql = (
        f"SELECT {ANOMALY_SUMMARY} FROM anomalies WHERE {' AND '.join(where)} "
        "ORDER BY first_interval_end, event_id LIMIT :limit"
    )
    with engine.connect() as conn:
        return list(conn.execute(text(sql), params).all())


def anomaly_event(engine: Engine, run_ids: Sequence[int], event_id: int) -> Any | None:
    with engine.connect() as conn:
        return conn.execute(
            text(
                f"SELECT {ANOMALY_SUMMARY}, evidence FROM anomalies "
                "WHERE event_id = :e AND run_id = ANY(:runs)"
            ),
            {"e": event_id, "runs": list(run_ids)},
        ).one_or_none()


def ping(engine: Engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
