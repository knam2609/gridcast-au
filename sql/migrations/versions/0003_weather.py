"""Weather sites, archived weather forecasts (variant C) and reanalysis observations (D).

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-03
"""

from __future__ import annotations

from gridcast.database.migrations import run_sql_file

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("0003_weather.up.sql")


def downgrade() -> None:
    run_sql_file("0003_weather.down.sql")
