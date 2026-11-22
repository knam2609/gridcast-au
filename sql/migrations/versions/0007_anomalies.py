"""Supply totals, anomaly events and anomaly runs: generation_intervals, anomalies (M4).

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-15
"""

from __future__ import annotations

from gridcast.database.migrations import run_sql_file

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("0007_anomalies.up.sql")


def downgrade() -> None:
    run_sql_file("0007_anomalies.down.sql")
