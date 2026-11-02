import datetime as dt
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
from sqlalchemy import Engine, text

from gridcast.config import AppConfig, ExpectedMissingConfig, load_config, load_market_limits
from gridcast.database.load import ensure_regions, start_ingestion_run, upsert_market_intervals
from gridcast.database.queries import load_weather_forecasts, load_weather_observations
from gridcast.ingestion.open_meteo import OpenMeteoClient
from gridcast.ingestion.weather import ingest_weather_range, month_hours
from gridcast.timeutils import month_bounds_utc
from gridcast.validation.checks import WEATHER_CHECKS, DQResult, run_checks
from gridcast.weather.align import site_weighted, to_five_minute
from tests.integration.factories import clean_month
from tests.open_meteo_fixtures import TODAY, FakeOpenMeteo, om_config

pytestmark = pytest.mark.integration

OCT = dt.date(2021, 10, 1)
N_SITES = 5
N_HOURS = 31 * 24 + 1  # NEM month of October plus the closing hour


def _client(fake: FakeOpenMeteo, tmp_path: Path) -> OpenMeteoClient:
    return OpenMeteoClient(om_config(), tmp_path, transport=fake.transport(), today=lambda: TODAY)


def _count(engine: Engine, sql: str) -> int:
    with engine.connect() as conn:
        return int(conn.execute(text(sql)).scalar_one())


def _sql(engine: Engine, sql: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(sql))


@pytest.fixture
def loaded(db: Engine, tmp_path: Path) -> Engine:
    """October 2021: clean market data plus fake weather for both datasets."""
    cfg = load_config()
    with db.begin() as conn:
        ensure_regions(conn, cfg)
        run = start_ingestion_run(conn, "mmsdm", "market_intervals", OCT)
        upsert_market_intervals(conn, clean_month(), run)
        conn.execute(text("UPDATE ingestion_runs SET status='success' WHERE run_id=:r"), {"r": run})
    results = ingest_weather_range(db, cfg, OCT, OCT, client=_client(FakeOpenMeteo(), tmp_path))
    assert [r.status for r in results] == ["success", "success"]
    return db


def _checks(engine: Engine, cfg: AppConfig | None = None) -> dict[str, DQResult]:
    cfg = cfg or load_config()
    results = run_checks(
        engine, cfg, load_market_limits(cfg), ["VIC1"], OCT, OCT, checks=WEATHER_CHECKS
    )
    return {r.check_name: r for r in results}


def test_ingest_loads_every_site_offset_and_hour(loaded: Engine) -> None:
    assert _count(loaded, "SELECT count(*) FROM weather_forecasts") == N_SITES * 2 * N_HOURS
    assert _count(loaded, "SELECT count(*) FROM weather_observations") == N_SITES * N_HOURS
    assert _count(loaded, "SELECT round(sum(weight)::numeric, 12) FROM weather_sites") == 1
    first, last = month_hours(OCT)
    with loaded.connect() as conn:
        span = conn.execute(
            text("SELECT min(valid_time), max(valid_time) FROM weather_observations")
        ).one()
        runs = conn.execute(
            text(
                "SELECT dataset, status, rows_loaded, files FROM ingestion_runs "
                "WHERE source = 'open_meteo' ORDER BY dataset"
            )
        ).all()
    assert (pd.Timestamp(span[0]), pd.Timestamp(span[1])) == (first, last)
    assert [(r.dataset, r.status) for r in runs] == [
        ("weather_forecasts", "success"),
        ("weather_observations", "success"),
    ]
    files: list[dict[str, Any]] = runs[0].files
    assert len(files) == N_SITES and files[0]["grid_latitude"] is not None


def test_reingest_uses_cache_and_changes_nothing(loaded: Engine, tmp_path: Path) -> None:
    fake = FakeOpenMeteo()
    results = ingest_weather_range(loaded, load_config(), OCT, OCT, client=_client(fake, tmp_path))
    assert fake.requests == []  # every site-month served from the cache
    assert [r.rows_loaded for r in results] == [0, 0]


def test_failed_month_is_recorded(db: Engine, tmp_path: Path) -> None:
    fake = FakeOpenMeteo()
    fake.responses = [httpx.Response(400, json={"error": True, "reason": "bad model"})]
    results = ingest_weather_range(
        db, load_config(), OCT, OCT, kinds=("forecasts",), client=_client(fake, tmp_path)
    )
    assert results[0].status == "failed" and "bad model" in (results[0].error or "")
    assert _count(db, "SELECT count(*) FROM weather_forecasts") == 0
    cfg = load_config()
    r = run_checks(db, cfg, load_market_limits(cfg), ["VIC1"], OCT, OCT)
    assert {x.check_name: x.status for x in r}["ingestion_failures"] == "fail"


def test_clean_weather_passes_every_check(loaded: Engine) -> None:
    results = _checks(loaded)
    assert set(results) == {c.name for c in WEATHER_CHECKS}
    assert {n: r.status for n, r in results.items() if r.status != "pass"} == {}


def test_unexpected_gap_fails_and_known_gap_warns(loaded: Engine) -> None:
    _sql(
        loaded,
        "UPDATE weather_forecasts SET temperature_2m = NULL WHERE site_id = 'geelong' "
        "AND offset_days = 1 "
        "AND valid_time BETWEEN '2021-10-10 00:00+00' AND '2021-10-10 02:00+00'",
    )
    _sql(
        loaded,
        "DELETE FROM weather_observations WHERE site_id = 'bendigo' "
        "AND valid_time = '2021-10-12 05:00+00'",
    )
    r = _checks(loaded)
    gaps = r["weather_gaps"]
    assert gaps.status == "fail"
    assert gaps.n_violations == 3 + 6  # 3 temperature hours + one missing row of 6 variables
    fc_gap = [e for e in gaps.details["examples"] if e["dataset"] == "forecast"]
    assert [(e["site_id"], e["offset_days"], e["n_missing"]) for e in fc_gap] == [("geelong", 1, 3)]

    # Declare the forecast gap as known: it moves from weather_gaps to weather_known_gaps.
    cfg = load_config()
    cfg.weather.forecasts.expected_missing = [
        ExpectedMissingConfig(
            reason="test_outage",
            variables=["temperature_2m"],
            offset_days=1,
            first=dt.datetime(2021, 10, 10, tzinfo=dt.UTC),
            last=dt.datetime(2021, 10, 10, 2, tzinfo=dt.UTC),
        )
    ]
    r = _checks(loaded, cfg)
    assert r["weather_gaps"].n_violations == 6
    assert r["weather_known_gaps"].status == "warn"
    assert r["weather_known_gaps"].n_violations == 3
    assert r["weather_known_gaps"].details["examples"][0]["known_reason"] == "test_outage"


def test_out_of_range_and_run_timing(loaded: Engine) -> None:
    _sql(
        loaded,
        "UPDATE weather_observations SET temperature_2m = 51 WHERE site_id = 'ballarat' "
        "AND valid_time = '2021-10-15 03:00+00'",
    )
    # Passes the table's generic constraint (available after run start) but not the rule.
    _sql(
        loaded,
        "UPDATE weather_forecasts SET available_at = run_init + interval '5 hours' "
        "WHERE site_id = 'melbourne' AND offset_days = 2 AND valid_time = '2021-10-15 03:00+00'",
    )
    r = _checks(loaded)
    assert r["weather_out_of_range"].n_violations == 1
    assert r["weather_out_of_range"].details["examples"][0]["variable"] == "temperature_2m"
    assert r["weather_run_timing"].n_violations == 1


def test_stale_weather_and_bad_weights(loaded: Engine) -> None:
    _, last_interval = month_bounds_utc(2021, 10)
    _sql(
        loaded,
        f"DELETE FROM weather_forecasts WHERE site_id = 'bendigo' AND offset_days = 2 "
        f"AND valid_time = '{last_interval.isoformat()}'",
    )
    _sql(loaded, "UPDATE weather_sites SET weight = weight * 2 WHERE site_id = 'geelong'")
    r = _checks(loaded)
    assert r["weather_stale"].status == "fail"
    assert r["weather_stale"].details["examples"][0]["site_id"] == "bendigo"
    assert r["weather_site_weights"].status == "fail"


def test_database_constraints_reject_bad_rows(loaded: Engine) -> None:
    with pytest.raises(Exception, match="weather_forecasts_run_chk"):
        _sql(
            loaded,
            "UPDATE weather_forecasts SET run_init = valid_time, "
            "available_at = valid_time + interval '6 hours' WHERE site_id = 'melbourne' "
            "AND offset_days = 1 AND valid_time = '2021-10-15 03:00+00'",
        )
    with pytest.raises(Exception, match="weather_observations_hour_chk"):
        _sql(
            loaded,
            "UPDATE weather_observations SET valid_time = valid_time + interval '30 minutes' "
            "WHERE site_id = 'melbourne' AND valid_time = '2021-10-31 14:00+00'",
        )


def test_loaded_weather_aligns_to_every_interval(loaded: Engine) -> None:
    cfg = load_config()
    first, last = month_hours(OCT)
    weights = cfg.weather.site_weights("VIC1")
    kinds = {v: c.kind for v, c in cfg.weather.variables.items()}
    targets = pd.DatetimeIndex(clean_month()["interval_end"])

    fc = load_weather_forecasts(loaded, "gfs_seamless", first, last)
    weighted = site_weighted(fc, weights, list(kinds), by=["offset_days"])
    aligned = to_five_minute(weighted, targets, kinds, by="offset_days")
    assert len(aligned) == 2 * len(targets) == 2 * 8928
    assert aligned[list(kinds)].notna().all().all()
    # Every interval's forecast became available after its run started, never after the
    # later bracketing hour's availability.
    assert (aligned["available_at"] >= aligned["interval_end"] - pd.Timedelta(hours=54)).all()

    obs = load_weather_observations(loaded, "era5", first, last)
    aligned_obs = to_five_minute(site_weighted(obs, weights, list(kinds)), targets, kinds)
    assert aligned_obs["temperature_2m"].notna().all()


def test_rounding_artefacts_are_clipped_and_reported(db: Engine, tmp_path: Path) -> None:
    def artefact(request: httpx.Request, payload: dict[str, Any]) -> dict[str, Any]:
        if request.url.params["models"] == "gfs_seamless":
            payload["hourly"]["cloud_cover_previous_day2"][100] = 101.0  # inside October
        return payload

    cfg = load_config()
    fake = FakeOpenMeteo(mutate=artefact)
    results = ingest_weather_range(db, cfg, OCT, OCT, client=_client(fake, tmp_path))
    assert [r.status for r in results] == ["success", "success"]
    assert _count(db, "SELECT max(cloud_cover) FROM weather_forecasts") <= 100
    r = run_checks(db, cfg, load_market_limits(cfg), ["VIC1"], OCT, OCT, checks=WEATHER_CHECKS)
    clipped = {x.check_name: x for x in r}["weather_clipped_values"]
    assert clipped.status == "warn"
    assert clipped.n_violations == N_SITES  # one artefact per site's forecast response
    assert clipped.details["examples"][0]["variable"] == "cloud_cover"
