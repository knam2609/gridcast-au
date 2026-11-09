"""Store DISPATCHPRICE APCFLAG / MARKETSUSPENDEDFLAG as raw codes + decoded booleans.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-02
"""

from __future__ import annotations

from gridcast.database.migrations import run_sql_file

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("0002_price_flag_codes.up.sql")


def downgrade() -> None:
    run_sql_file("0002_price_flag_codes.down.sql")
