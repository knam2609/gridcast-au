"""Integration fixtures: a dedicated, migrated test database on the compose PostgreSQL."""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, make_url, text

from gridcast.config import PROJECT_ROOT, get_settings, load_config
from gridcast.database.load import ensure_regions, start_ingestion_run, upsert_market_intervals
from gridcast.ingestion.open_meteo import OpenMeteoClient
from gridcast.ingestion.weather import ingest_weather_range
from tests.integration.factories import clean_month
from tests.open_meteo_fixtures import TODAY, FakeOpenMeteo, om_config

OCT = dt.date(2021, 10, 1)

TABLES = (
    "dq_results, forecasts, forecast_metrics, model_runs, calendar_days, "
    "weather_forecasts, weather_observations, weather_sites, "
    "market_intervals, ingestion_runs, regions"
)


def _test_url() -> str:
    settings = get_settings()
    if settings.test_database_url:
        return settings.test_database_url
    url = make_url(settings.database_url)
    return url.set(database=f"{url.database}_test").render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    url = make_url(_test_url())
    if not url.database or not url.database.endswith("_test"):
        pytest.exit(f"Refusing to run integration tests against non-test database {url.database}")
    admin = create_engine(
        url.set(database="postgres"), isolation_level="AUTOCOMMIT", pool_pre_ping=True
    )
    try:
        with admin.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :d"), {"d": url.database}
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    except Exception as exc:  # pragma: no cover - environment problem
        pytest.skip(f"PostgreSQL not reachable ({exc}); run `make db-up`")
    finally:
        admin.dispose()

    rendered = url.render_as_string(hide_password=False)
    previous = os.environ.get("GRIDCAST_DATABASE_URL")
    os.environ["GRIDCAST_DATABASE_URL"] = rendered
    try:
        command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")
    finally:
        if previous is None:
            os.environ.pop("GRIDCAST_DATABASE_URL", None)
        else:
            os.environ["GRIDCAST_DATABASE_URL"] = previous

    eng = create_engine(rendered, connect_args={"options": "-c timezone=UTC"})
    yield eng
    eng.dispose()


@pytest.fixture
def db(engine: Engine) -> Engine:
    """Empty all application tables before each test."""
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {TABLES} RESTART IDENTITY CASCADE"))
    return engine


def _client(fake: FakeOpenMeteo, tmp_path: Path) -> OpenMeteoClient:
    return OpenMeteoClient(om_config(), tmp_path, transport=fake.transport(), today=lambda: TODAY)


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
