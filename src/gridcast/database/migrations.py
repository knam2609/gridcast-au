"""Helpers for Alembic migrations whose bodies are plain SQL files in ``sql/schema``."""

from __future__ import annotations

from alembic import op

from gridcast.config import PROJECT_ROOT

SCHEMA_DIR = PROJECT_ROOT / "sql" / "schema"


def run_sql_file(name: str) -> None:
    """Execute a SQL file verbatim inside the current migration transaction.

    ``exec_driver_sql`` bypasses SQLAlchemy's bind-parameter parsing so the SQL is sent
    exactly as written (multi-statement scripts are fine with psycopg when no parameters
    are passed).
    """
    sql = (SCHEMA_DIR / name).read_text()
    op.get_bind().exec_driver_sql(sql)
