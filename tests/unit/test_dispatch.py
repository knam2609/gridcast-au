import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from gridcast.ingestion.dispatch import (
    DISPATCHPRICE,
    DISPATCHREGIONSUM,
    build_market_intervals,
    extract_table,
)
from gridcast.ingestion.mms_csv import MmsFormatError
from tests.mms_fixtures import (
    PRICE_COLS,
    REGIONSUM_COLS,
    mms_lines,
    nem,
    price_rows,
    regionsum_rows,
    write_zip,
)

START = dt.datetime(2021, 10, 1, 0, 5)


def _regionsum(tmp_path: Path, rows: list[dict[str, str]], name: str = "drs.zip") -> Path:
    lines = mms_lines("DISPATCH", "REGIONSUM", "5", REGIONSUM_COLS, rows)
    return write_zip(tmp_path / name, "DRS.CSV", lines)


def _price(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    return write_zip(
        tmp_path / "dp.zip", "DP.CSV", mms_lines("DISPATCH", "PRICE", "4", PRICE_COLS, rows)
    )


def test_nem_time_converted_to_utc_and_interval_start_derived(tmp_path: Path) -> None:
    ext = extract_table(
        [_regionsum(tmp_path, regionsum_rows(START, 2))], DISPATCHREGIONSUM, ["VIC1"]
    )
    first = ext.frame.iloc[0]
    # 2021-10-01 00:05 NEM time (UTC+10, no DST) == 2021-09-30 14:05 UTC
    assert first["interval_end"] == pd.Timestamp("2021-09-30 14:05", tz="UTC")
    assert first["interval_start"] == pd.Timestamp("2021-09-30 14:00", tz="UTC")
    assert first["region_id"] == "VIC1"
    assert first["total_demand_mw"] == pytest.approx(5000.0)


def test_nem_time_ignores_melbourne_dst(tmp_path: Path) -> None:
    # 2022-01-15 is in AEDT (UTC+11) locally, but NEM time is still UTC+10.
    rows = regionsum_rows(dt.datetime(2022, 1, 15, 12, 0), 1, regions=("VIC1",))
    ext = extract_table([_regionsum(tmp_path, rows)], DISPATCHREGIONSUM, ["VIC1"])
    assert ext.frame["interval_end"].iloc[0] == pd.Timestamp("2022-01-15 02:00", tz="UTC")


def test_duplicates_keep_latest_lastchanged(tmp_path: Path) -> None:
    rows = regionsum_rows(START, 2, regions=("VIC1",))
    revised = dict(rows[0], TOTALDEMAND="4321.00", LASTCHANGED=nem(START + dt.timedelta(hours=1)))
    stale = dict(rows[0], TOTALDEMAND="1.00", LASTCHANGED=nem(START - dt.timedelta(hours=1)))
    ext = extract_table(
        [_regionsum(tmp_path, [revised, *rows, stale])], DISPATCHREGIONSUM, ["VIC1"]
    )
    assert ext.rows_parsed == 4
    assert ext.duplicates_dropped == 2
    assert len(ext.frame) == 2
    assert ext.frame["total_demand_mw"].iloc[0] == pytest.approx(4321.0)


def test_intervention_rows_dropped(tmp_path: Path) -> None:
    rows = regionsum_rows(START, 2, regions=("VIC1",))
    intervention = dict(rows[0], INTERVENTION="1", TOTALDEMAND="9999")
    ext = extract_table([_regionsum(tmp_path, [*rows, intervention])], DISPATCHREGIONSUM, ["VIC1"])
    assert ext.intervention_rows_dropped == 1
    assert 9999.0 not in ext.frame["total_demand_mw"].tolist()


def test_non_numeric_value_raises(tmp_path: Path) -> None:
    rows = regionsum_rows(START, 1, regions=("VIC1",))
    rows[0]["TOTALDEMAND"] = "abc"
    with pytest.raises(MmsFormatError, match="non-numeric"):
        extract_table([_regionsum(tmp_path, rows)], DISPATCHREGIONSUM, ["VIC1"])


def test_blank_numeric_becomes_nan(tmp_path: Path) -> None:
    rows = regionsum_rows(START, 1, regions=("VIC1",))
    rows[0]["NETINTERCHANGE"] = ""
    ext = extract_table([_regionsum(tmp_path, rows)], DISPATCHREGIONSUM, ["VIC1"])
    assert ext.frame["net_interchange_mw"].isna().all()


def test_multipart_files_are_concatenated(tmp_path: Path) -> None:
    p1 = _regionsum(tmp_path, regionsum_rows(START, 2, regions=("VIC1",)), "p1.zip")
    later = START + dt.timedelta(minutes=10)
    p2 = _regionsum(tmp_path, regionsum_rows(later, 2, regions=("VIC1",)), "p2.zip")
    ext = extract_table([p1, p2], DISPATCHREGIONSUM, ["VIC1"])
    assert len(ext.frame) == 4
    assert ext.frame["interval_end"].is_monotonic_increasing


def test_build_market_intervals_outer_join_keeps_unmatched(tmp_path: Path) -> None:
    demand = extract_table(
        [_regionsum(tmp_path, regionsum_rows(START, 3, regions=("VIC1",)))],
        DISPATCHREGIONSUM,
        ["VIC1"],
    )
    price = extract_table(
        [_price(tmp_path, price_rows(START, 2, regions=("VIC1",)))], DISPATCHPRICE, ["VIC1"]
    )
    merged = build_market_intervals(demand.frame, price.frame)
    assert len(merged) == 3
    assert merged["rrp_aud_mwh"].isna().sum() == 1
    assert merged["apc_flag"].dtype == bool
    assert not merged["market_suspended_flag"].any()
