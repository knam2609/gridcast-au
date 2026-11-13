"""AEMO probabilistic (POE) demand forecasts: aemo_demand_poe (M3).

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-13
"""

from __future__ import annotations

from gridcast.database.migrations import run_sql_file

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("0006_aemo_poe.up.sql")


def downgrade() -> None:
    run_sql_file("0006_aemo_poe.down.sql")
