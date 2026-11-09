"""Core schema: regions, market intervals, lineage, calendar, models, forecasts, DQ.

Revision ID: 0001
Revises:
Create Date: 2026-11-02
"""

from __future__ import annotations

from gridcast.database.migrations import run_sql_file

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("0001_core.up.sql")


def downgrade() -> None:
    run_sql_file("0001_core.down.sql")
