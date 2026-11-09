"""Integration fixtures: a dedicated, migrated test database on the compose PostgreSQL."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, make_url, text

from gridcast.config import PROJECT_ROOT, get_settings

TABLES = (
    "dq_results, forecasts, forecast_metrics, model_runs, calendar_days, "
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
