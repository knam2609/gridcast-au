"""Schema validation (pandera) for frames before they are loaded into PostgreSQL.

Schema checks reject *structurally* invalid data (wrong types, off-grid timestamps,
duplicate keys, values outside the market's hard limits). Softer "is this plausible?"
questions are answered by the data-quality checks in :mod:`gridcast.validation.checks`,
which flag rather than reject.
"""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd
import pandera.pandas as pa

from gridcast.timeutils import INTERVAL

MARKET_FLOOR = -1000.0
RRP_SANITY_MAX = 100_000.0  # mirrors the DB CHECK constraint; true caps are per-FY DQ checks


def _is_utc(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.DatetimeTZDtype) and str(s.dtype.tz) == "UTC"


def _on_grid(s: pd.Series) -> pd.Series:
    return s.dt.floor(INTERVAL) == s


utc_timestamp = [
    pa.Check(_is_utc, element_wise=False, error="must be tz-aware UTC"),
    pa.Check(_on_grid, error="must be on the 5-minute grid"),
]


def market_intervals_schema(regions: Collection[str]) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "region_id": pa.Column(str, pa.Check.isin(sorted(regions)), coerce=True),
            "interval_start": pa.Column(checks=utc_timestamp, nullable=False),
            "interval_end": pa.Column(checks=utc_timestamp, nullable=False),
            "total_demand_mw": pa.Column(
                float, pa.Check(lambda s: pd.Series(abs(s) < 1e6), error="finite"), nullable=True
            ),
            "rrp_aud_mwh": pa.Column(
                float, pa.Check.in_range(MARKET_FLOOR, RRP_SANITY_MAX), nullable=True
            ),
            "net_interchange_mw": pa.Column(float, nullable=True),
            "available_generation_mw": pa.Column(float, nullable=True),
            "apc_flag_bits": pa.Column("int16", pa.Check.in_range(0, 31)),
            "market_suspended_flag": pa.Column("int16", pa.Check.ge(0)),
            "price_status": pa.Column(str, nullable=True, coerce=True),
        },
        checks=[
            pa.Check(
                lambda df: (df["interval_end"] - df["interval_start"]) == INTERVAL,
                error="interval_end - interval_start must equal 5 minutes",
            ),
            pa.Check(
                lambda df: df["total_demand_mw"].notna() | df["rrp_aud_mwh"].notna(),
                error="row has neither demand nor price",
            ),
            pa.Check(
                lambda df: ~df.duplicated(["region_id", "interval_end"], keep=False),
                error="duplicate (region_id, interval_end)",
            ),
        ],
        strict=True,
        ordered=False,
        name="market_intervals",
    )


def validate_market_intervals(df: pd.DataFrame, regions: Collection[str]) -> pd.DataFrame:
    """Validate (lazily, collecting every failure) and return the validated frame."""
    return market_intervals_schema(regions).validate(df.reset_index(drop=True), lazy=True)


def summarise_schema_errors(exc: pa.errors.SchemaErrors, max_checks: int = 10) -> str:
    """Compact one-line-per-check summary of a lazy validation failure."""
    fc = exc.failure_cases
    grouped = (
        fc.groupby(["column", "check"], dropna=False)["index"]
        .agg(n="nunique", first_index="min")
        .reset_index()
        .head(max_checks)
    )
    parts = [
        f"{row.column}: {row.check} (rows={row.n}, first index={row.first_index})"
        for row in grouped.itertuples()
    ]
    return "; ".join(parts)
