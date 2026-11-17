"""Detector inputs from the database (M4 step 2b): the market/supply join and day types."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine, text

from gridcast.anomalies.inputs import load_market_series
from gridcast.config import load_config
from gridcast.database.load import (
    ensure_regions,
    start_ingestion_run,
    upsert_generation,
    upsert_market_intervals,
)
from gridcast.features.calendar import build_calendar_days, upsert_calendar_days
from gridcast.timeutils import INTERVAL
from tests.integration.factories import clean_month

pytestmark = pytest.mark.integration

FEB = dt.date(2022, 2, 1)


def _generation(market: pd.DataFrame, kinds: tuple[str, ...]) -> pd.DataFrame:
    n = len(market)
    parts = []
    for k, kind in enumerate(kinds):
        cleared = np.linspace(100, 900, n) + 10 * k
        parts.append(
            pd.DataFrame(
                {
                    "region_id": "VIC1",
                    "interval_end": market["interval_end"],
                    "kind": kind,
                    "cleared_mw": cleared,
                    "uigf_mw": cleared + 50 if kind in ("wind", "solar") else np.nan,
                    "available_mw": np.nan,
                    "load_mw": np.nan,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _load(db: Engine, kinds: tuple[str, ...] = ("wind", "solar", "dispatchable")) -> pd.DataFrame:
    cfg = load_config()
    region = cfg.region("VIC1")
    market = clean_month(2022, 2)
    with db.begin() as conn:
        ensure_regions(conn, cfg)
        run = start_ingestion_run(conn, "mmsdm", "market_intervals", FEB)
        upsert_market_intervals(conn, market, run)
        upsert_generation(conn, _generation(market, kinds), run)
        upsert_calendar_days(
            conn,
            build_calendar_days(
                region.holiday_subdiv, region.local_tz, dt.date(2022, 1, 31), dt.date(2022, 3, 2)
            ),
        )
    return market


def test_market_series_joins_supply_and_day_types(db: Engine) -> None:
    market = _load(db)
    series = load_market_series(db, load_config(), "VIC1")
    assert len(series) == len(market) == 28 * 288
    np.testing.assert_array_equal(series.demand, market["total_demand_mw"].to_numpy())
    np.testing.assert_array_equal(series.price, market["rrp_aud_mwh"].to_numpy())
    np.testing.assert_allclose(series.wind_uigf - series.wind_cleared, 50.0)
    np.testing.assert_allclose(series.solar_cleared - series.wind_cleared, 10.0)
    # February 2022 has no Victorian public holiday: business days are the weekdays.
    local = (series.ends - INTERVAL).tz_convert("Australia/Melbourne")
    np.testing.assert_array_equal(series.business, np.asarray(local.day_of_week < 5))


def test_market_series_stops_at_last_end(db: Engine) -> None:
    market = _load(db)
    last = market["interval_end"].iloc[287]
    series = load_market_series(db, load_config(), "VIC1", last_end=last)
    assert len(series) == 288 and series.ends[-1] == last


def test_missing_supply_rows_are_refused(db: Engine) -> None:
    _load(db, kinds=("wind", "dispatchable"))
    with pytest.raises(ValueError, match="solar_cleared"):
        load_market_series(db, load_config(), "VIC1")


def test_missing_calendar_days_are_refused(db: Engine) -> None:
    _load(db)
    with db.begin() as conn:
        conn.execute(text("DELETE FROM calendar_days WHERE local_date = '2022-02-14'"))
    with pytest.raises(LookupError, match="calendar_days"):
        load_market_series(db, load_config(), "VIC1")
