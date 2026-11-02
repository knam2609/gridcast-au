"""Extract, standardise and deduplicate AEMO dispatch tables into ``market_intervals`` rows."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from gridcast.ingestion.mms_csv import MmsFormatError, ReportKey, read_mms_report
from gridcast.timeutils import INTERVAL, parse_nem_timestamps

KEY_COLUMNS = ("SETTLEMENTDATE", "RUNNO", "REGIONID", "INTERVENTION", "LASTCHANGED")
MARKET_KEY = ["region_id", "interval_start", "interval_end"]


@dataclass(frozen=True, slots=True)
class TableSpec:
    """Which columns to keep from an MMS table and what to call them."""

    name: str
    key: ReportKey
    numeric: dict[str, str] = field(default_factory=dict)
    flags: dict[str, str] = field(default_factory=dict)
    text: dict[str, str] = field(default_factory=dict)

    @property
    def source_columns(self) -> list[str]:
        return [*KEY_COLUMNS, *self.numeric, *self.flags, *self.text]

    @property
    def output_columns(self) -> list[str]:
        return [*self.numeric.values(), *self.flags.values(), *self.text.values()]


DISPATCHREGIONSUM = TableSpec(
    name="DISPATCHREGIONSUM",
    key=ReportKey("DISPATCH", "REGIONSUM"),
    numeric={
        "TOTALDEMAND": "total_demand_mw",
        "NETINTERCHANGE": "net_interchange_mw",
        "AVAILABLEGENERATION": "available_generation_mw",
    },
)

DISPATCHPRICE = TableSpec(
    name="DISPATCHPRICE",
    key=ReportKey("DISPATCH", "PRICE"),
    numeric={"RRP": "rrp_aud_mwh"},
    flags={"APCFLAG": "apc_flag", "MARKETSUSPENDEDFLAG": "market_suspended_flag"},
    text={"PRICE_STATUS": "price_status"},
)

TABLE_SPECS: dict[str, TableSpec] = {s.name: s for s in (DISPATCHREGIONSUM, DISPATCHPRICE)}


@dataclass(slots=True)
class TableExtract:
    """Standardised rows for one table-month plus lineage statistics."""

    spec: TableSpec
    frame: pd.DataFrame
    rows_parsed: int
    intervention_rows_dropped: int
    duplicates_dropped: int
    versions: set[str]


def _to_float(values: pd.Series, name: str) -> pd.Series:
    """Strict numeric conversion: blanks become NaN, anything else unparseable raises."""
    stripped = values.str.strip()
    out = pd.to_numeric(stripped.mask(stripped.eq("")), errors="coerce").astype("float64")
    bad = out.isna() & stripped.ne("").fillna(False)
    if bad.any():
        sample = stripped[bad].head(3).tolist()
        raise MmsFormatError(f"{name}: {int(bad.sum())} non-numeric values, e.g. {sample}")
    return out


def _to_flag(values: pd.Series, name: str) -> pd.Series:
    stripped = values.str.strip().fillna("")
    unknown = ~stripped.isin(["0", "1", ""])
    if unknown.any():
        raise MmsFormatError(f"{name}: unexpected flag values {stripped[unknown].unique()[:5]}")
    return stripped.eq("1")


def standardise(raw: pd.DataFrame, spec: TableSpec) -> tuple[pd.DataFrame, int, int]:
    """Filter the pricing run, convert types/timezones, and deduplicate.

    Returns ``(frame, intervention_rows_dropped, duplicates_dropped)``.
    Deduplication keeps the most recently written record (``LASTCHANGED``, then ``RUNNO``)
    for each ``(region, interval_end)``.
    """
    is_pricing_run = raw["INTERVENTION"].str.strip().eq("0")
    intervention_dropped = int((~is_pricing_run).sum())
    df = raw.loc[is_pricing_run]

    out = pd.DataFrame(
        {
            "region_id": df["REGIONID"].str.strip().astype("string"),
            "interval_end": parse_nem_timestamps(df["SETTLEMENTDATE"]),
            "_lastchanged": parse_nem_timestamps(df["LASTCHANGED"]),
            "_runno": pd.to_numeric(df["RUNNO"], errors="raise").astype("int64"),
        }
    )
    out["interval_start"] = out["interval_end"] - INTERVAL
    for src, dest in spec.numeric.items():
        out[dest] = _to_float(df[src], f"{spec.name}.{src}")
    for src, dest in spec.flags.items():
        out[dest] = _to_flag(df[src], f"{spec.name}.{src}")
    for src, dest in spec.text.items():
        text = df[src].str.strip()
        out[dest] = text.mask(text.eq("")).astype("string")

    before = len(out)
    out = (
        out.sort_values(["region_id", "interval_end", "_lastchanged", "_runno"])
        .drop_duplicates(subset=["region_id", "interval_end"], keep="last")
        .drop(columns=["_lastchanged", "_runno"])
        .reset_index(drop=True)
    )
    return out[[*MARKET_KEY, *spec.output_columns]], intervention_dropped, before - len(out)


def extract_table(paths: Sequence[Path], spec: TableSpec, regions: Collection[str]) -> TableExtract:
    """Parse and standardise all parts of one table-month."""
    frames: list[pd.DataFrame] = []
    versions: set[str] = set()
    for path in paths:
        frame, stats = read_mms_report(path, spec.key, spec.source_columns, region_filter=regions)
        frames.append(frame)
        versions |= stats.versions
    raw = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=spec.source_columns)
    )
    clean, n_intervention, n_dupes = standardise(raw, spec)
    return TableExtract(
        spec=spec,
        frame=clean,
        rows_parsed=len(raw),
        intervention_rows_dropped=n_intervention,
        duplicates_dropped=n_dupes,
        versions=versions,
    )


def build_market_intervals(demand: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """Outer-join demand and price on the interval key.

    An outer join keeps intervals present in only one table (with nulls) so the
    data-quality checks can report them instead of silently losing rows.
    """
    merged = demand.merge(price, on=MARKET_KEY, how="outer", validate="one_to_one")
    for flag in DISPATCHPRICE.flags.values():
        merged[flag] = merged[flag].astype("boolean").fillna(False).astype(bool)
    return merged.sort_values(["region_id", "interval_end"]).reset_index(drop=True)
