"""Alembic environment.

Migrations are hand-written SQL (sql/schema/*.sql) executed via ``run_sql_file``; no ORM
metadata or autogenerate is used. The URL comes from GridCast settings (env vars).
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from gridcast.config import get_settings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None


def _database_url() -> str:
    # `-x db_url=...` (used by integration tests) overrides the configured URL.
    return context.get_x_argument(as_dictionary=True).get("db_url") or os.environ.get(
        "GRIDCAST_DATABASE_URL", get_settings().database_url
    )


def run_migrations_offline() -> None:
    context.configure(url=_database_url(), literal_binds=True, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
