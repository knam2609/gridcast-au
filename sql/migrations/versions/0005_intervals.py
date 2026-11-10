"""Interval batches: model_runs rows of kind 'interval' (M3).

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-10
"""

from __future__ import annotations

from gridcast.database.migrations import run_sql_file

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("0005_intervals.up.sql")


def downgrade() -> None:
    run_sql_file("0005_intervals.down.sql")
