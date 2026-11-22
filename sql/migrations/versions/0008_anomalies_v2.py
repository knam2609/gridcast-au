"""M4 v2: the anomalies table accepts v2's D6 detector, 'robust_distance'.

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-20
"""

from __future__ import annotations

from gridcast.database.migrations import run_sql_file

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("0008_anomalies_v2.up.sql")


def downgrade() -> None:
    run_sql_file("0008_anomalies_v2.down.sql")
