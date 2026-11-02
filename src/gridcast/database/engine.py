"""SQLAlchemy engine construction (Core only; no ORM)."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine

from gridcast.config import get_settings


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    """Return a pooled engine for ``url`` (defaults to ``GRIDCAST_DATABASE_URL``).

    Sessions run in UTC so ``timestamptz`` values are rendered unambiguously.
    """
    return create_engine(
        url or get_settings().database_url,
        pool_pre_ping=True,
        connect_args={"options": "-c timezone=UTC"},
    )
