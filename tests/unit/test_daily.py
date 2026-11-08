"""NEMWeb daily dispatch reports: nested-zip parsing, listing, cache, day checks, comparison."""

import datetime as dt
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from gridcast.config import MmsdmSourceConfig
from gridcast.ingestion.daily import (
    VALUE_COLUMNS,
    compare_frames,
    day_interval_ends,
    month_days,
    transform_day,
)
from gridcast.ingestion.dispatch import DISPATCHREGIONSUM
from gridcast.ingestion.mms_csv import MmsFormatError, read_mms_report
from gridcast.ingestion.nemweb import NemwebClient, parse_daily_listing
from tests.mms_fixtures import price_rows, regionsum_rows, write_daily_archive

LISTING_URL = "https://nemweb.example/Reports/Archive/DispatchIS_Reports/"
DAY = dt.date(2026, 10, 1)
START = dt.datetime(2026, 10, 1, 0, 5)


def _cfg() -> MmsdmSourceConfig:
    return MmsdmSourceConfig(
        base_url="https://nemweb.example/MMSDM",
        user_agent="gridcast-test",
        request_delay_seconds=0.0,
        timeout_seconds=5,
        max_retries=2,
        tables=["DISPATCHREGIONSUM"],
    )


def _archive(path: Path, n: int = 288, chunks: int = 3) -> Path:
    return write_daily_archive(path, regionsum_rows(START, n), price_rows(START, n), chunks=chunks)


def test_reads_every_report_inside_a_zip_of_zips(tmp_path: Path) -> None:
    path = _archive(tmp_path / "PUBLIC_DISPATCHIS_20261001.zip")
    frame, stats = read_mms_report(
        path, DISPATCHREGIONSUM.key, DISPATCHREGIONSUM.source_columns, region_filter={"VIC1"}
    )
    assert stats.members == 3
    assert len(frame) == 288
    assert frame["SETTLEMENTDATE"].iloc[0] == "2026/10/01 00:05:00"
    assert frame["SETTLEMENTDATE"].iloc[-1] == "2026/10/02 00:00:00"


def test_truncated_inner_report_is_detected(tmp_path: Path) -> None:
    good = _archive(tmp_path / "good.zip", chunks=2)
    bad = tmp_path / "PUBLIC_DISPATCHIS_20261001.zip"
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(bad, "w") as dst:
        for name in src.namelist():
            blob = src.read(name)
            if name == sorted(src.namelist())[-1]:
                with zipfile.ZipFile(__import__("io").BytesIO(blob)) as inner:
                    csv_name = inner.namelist()[0]
                    text = inner.read(csv_name).decode("latin-1")
                cut = "\r\n".join(text.split("\r\n")[:-3]) + "\r\n"  # drop the footer
                buf = __import__("io").BytesIO()
                with zipfile.ZipFile(buf, "w") as rebuilt:
                    rebuilt.writestr(csv_name, cut)
                blob = buf.getvalue()
            dst.writestr(name, blob)
    with pytest.raises(MmsFormatError, match="END OF REPORT"):
        read_mms_report(bad, DISPATCHREGIONSUM.key, DISPATCHREGIONSUM.source_columns)


def test_day_bounds_are_nem_time() -> None:
    ends = day_interval_ends(DAY)
    assert len(ends) == 288
    assert ends[0] == pd.Timestamp("2026-09-30 14:05", tz="UTC")  # 00:05 NEM
    assert ends[-1] == pd.Timestamp("2026-10-01 14:00", tz="UTC")  # 24:00 NEM
    assert len(month_days(dt.date(2026, 9, 1))) == 30
    assert month_days(dt.date(2024, 2, 1))[-1] == dt.date(2024, 2, 29)


def test_transform_day_requires_the_complete_day(tmp_path: Path) -> None:
    full = transform_day(_archive(tmp_path / "a" / "PUBLIC_DISPATCHIS_20261001.zip"), DAY, ["VIC1"])
    assert len(full[0]) == 288
    short = _archive(tmp_path / "b" / "PUBLIC_DISPATCHIS_20261001.zip", n=287)
    with pytest.raises(ValueError, match="287 intervals"):
        transform_day(short, DAY, ["VIC1"])
    with pytest.raises(ValueError, match="expected the 288"):  # wrong day for the file
        transform_day(short, dt.date(2026, 10, 2), ["VIC1"])


def test_parse_daily_listing() -> None:
    html = """<pre>
    <A HREF="/Reports/ARCHIVE/DispatchIS_Reports/PUBLIC_DISPATCHIS_20260930.zip">a</A>
    <A HREF="/Reports/ARCHIVE/DispatchIS_Reports/PUBLIC_DISPATCHIS_20261001.zip">b</A>
    <A HREF="/Reports/ARCHIVE/DispatchIS_Reports/PUBLIC_DISPATCHIS_2026100100.zip">c</A>
    <A HREF="/Reports/ARCHIVE/Other/PUBLIC_TRADINGIS_20261001.zip">d</A>
    </pre>"""
    files = parse_daily_listing(html, LISTING_URL)
    assert sorted(files) == [dt.date(2026, 9, 30), DAY]
    f = files[DAY]
    assert (
        f.url
        == "https://nemweb.example/Reports/ARCHIVE/DispatchIS_Reports/PUBLIC_DISPATCHIS_20261001.zip"
    )
    assert (f.source_dir, f.dataset, f.part) == ("nemweb_daily", "DISPATCHIS", 1)


def test_fetch_daily_reports_caches_and_skips_the_listing(tmp_path: Path) -> None:
    calls: list[str] = []
    listing = '<A HREF="/Reports/ARCHIVE/DispatchIS_Reports/PUBLIC_DISPATCHIS_20261001.zip">x</A>'

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if str(request.url) == LISTING_URL:
            return httpx.Response(200, text=listing)
        return httpx.Response(200, content=b"zip-bytes")

    with NemwebClient(_cfg(), tmp_path, transport=httpx.MockTransport(handler)) as client:
        got = client.fetch_daily_reports(LISTING_URL, [DAY])
        again = client.fetch_daily_reports(LISTING_URL, [DAY])
        with pytest.raises(FileNotFoundError, match="2026-10-02"):
            client.fetch_daily_reports(LISTING_URL, [dt.date(2026, 10, 2)])
    assert got[DAY].path == (
        tmp_path / "nemweb_daily" / "DISPATCHIS" / "202610" / "PUBLIC_DISPATCHIS_20261001.zip"
    )
    assert again[DAY].from_cache
    assert len(calls) == 3  # listing + file, then (cached) nothing, then one listing


def _frame(values: dict[str, list[object]]) -> pd.DataFrame:
    ends = pd.date_range("2026-09-30 14:05", periods=3, freq="5min", tz="UTC")
    base: dict[str, object] = {
        "region_id": ["VIC1"] * 3,
        "interval_end": ends,
        "total_demand_mw": [5000.0, 5100.5, 5200.25],
        "rrp_aud_mwh": [50.0, np.nan, 70.0],
        "net_interchange_mw": [1.0, 2.0, 3.0],
        "available_generation_mw": [9000.0, 9000.0, 9000.0],
        "apc_flag_bits": np.array([0, 0, 0], dtype="int16"),
        "market_suspended_flag": np.array([0, 0, 0], dtype="int16"),
        "price_status": pd.array(["FIRM", None, "FIRM"], dtype="string"),
    }
    return pd.DataFrame(base | values)


def test_compare_frames_counts_exact_differences() -> None:
    daily = _frame({})
    same, only_d, only_l = compare_frames(daily, _frame({}))
    assert (only_d, only_l) == (0, 0)
    assert [c.column for c in same] == list(VALUE_COLUMNS)
    assert all(c.differing == 0 and c.compared == 3 for c in same)

    loaded = _frame(
        {
            "total_demand_mw": [5000.0, 5100.5, 5200.26],
            "rrp_aud_mwh": [50.0, 60.0, 70.0],  # missing vs present differs
            "price_status": pd.array(["FIRM", None, "NOT FIRM"], dtype="string"),
        }
    ).iloc[1:]
    cols, only_d, only_l = compare_frames(daily, loaded)
    by = {c.column: c for c in cols}
    assert (only_d, only_l) == (1, 0)
    assert by["total_demand_mw"].differing == 1 and by["total_demand_mw"].must_match
    assert by["total_demand_mw"].max_abs_difference == pytest.approx(0.01)
    assert by["rrp_aud_mwh"].differing == 1
    assert by["price_status"].differing == 1 and by["price_status"].max_abs_difference is None
    assert by["net_interchange_mw"].differing == 0
