"""AEMO pre-dispatch and 5-minute pre-dispatch demand forecasts, every run (M2 benchmark).

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-08
"""

from __future__ import annotations

from gridcast.database.migrations import run_sql_file

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("0004_aemo_forecasts.up.sql")


def downgrade() -> None:
    run_sql_file("0004_aemo_forecasts.down.sql")
