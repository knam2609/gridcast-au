"""M5: serving tables, forecast_intervals and publications (docs/plans/m5_plan.md §2.2).

Revision ID: 0009
Revises: 0008
Create Date: 2026-11-23
"""

from __future__ import annotations

from gridcast.database.migrations import run_sql_file

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("0009_serving.up.sql")


def downgrade() -> None:
    run_sql_file("0009_serving.down.sql")
